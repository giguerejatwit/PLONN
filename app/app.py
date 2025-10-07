import json
import os
import secrets
import time
from urllib.parse import urlencode

import requests
import stripe
from dotenv import load_dotenv
from flask import (Flask, make_response, redirect, render_template, request,
                   send_from_directory, session, url_for)

load_dotenv()

# --- Config from .env ---
STRIPE_SECRET = os.getenv("STRIPE_SECRET")
PRICE_ID = os.getenv("PRICE_ID")
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")
CODE_TTL_SECONDS = int(os.getenv("CODE_TTL_SECONDS", "900"))  # 15 minutes by default
DEV_DEBUG = os.getenv("DEV_DEBUG") == "1"

# --- Discord OAuth config ---
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET", "")
DISCORD_REDIRECT_PATH = "/oauth/discord/callback"
DISCORD_REDIRECT_URI = os.getenv("DISCORD_REDIRECT_URI", f"{BASE_URL}{DISCORD_REDIRECT_PATH}")

# Flask session secret (required for OAuth state)
APP_SECRET = os.getenv("FLASK_SECRET", "")

if not STRIPE_SECRET:
  raise RuntimeError("Missing STRIPE_SECRET in .env")
if not PRICE_ID:
  raise RuntimeError("Missing PRICE_ID in .env")

stripe.api_key = STRIPE_SECRET

TOKENS_FILE = "tokens.json"   # { code: { "stripe_customer_id": "...", "exp": 123456, "promoter_code": "..." } }
LINKS_FILE  = "links.json"    # { discord_user_id: "stripe_customer_id" }


app = Flask(__name__, static_folder="analytics", template_folder="analytics")
app.secret_key = APP_SECRET or os.urandom(24)

# --- Analytics HTML serving ---


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mlb")
def mlb():
    return render_template("mlb.html")

@app.route("/nba")
def nba():
    return render_template("nba.html")

@app.route("/analytics/<path:filename>")
def analytics_static(filename):
    return send_from_directory("analytics", filename)

# --- helpers ---
def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

@app.get("/join")
def join():
    # optional promoter code in query: ?ref=mingosjce
    ref = (request.args.get("ref") or "").strip().lower()
    resp = make_response("", 302)
    resp.headers["Location"] = "/api/checkout"
    if ref:
        resp.set_cookie("plonn_ref", ref, max_age=60*60*24*30, httponly=True, samesite="Lax")
    return resp

@app.get("/<ref>")
def direct_referral(ref):
    ref = ref.strip().lower()
    resp = make_response("", 302)
    resp.headers["Location"] = "/api/checkout"
    resp.set_cookie("plonn_ref", ref, max_age=60*60*24*30, httponly=True, samesite="Lax")
    return resp

@app.route("/api/checkout", methods=["GET", "POST"])
def api_checkout():
    ref = request.cookies.get("plonn_ref", "")
    # Create Stripe Checkout session for a subscription
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": PRICE_ID, "quantity": 1}],
        success_url=f"{BASE_URL}/success?sid={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{BASE_URL}/cancel",
        # NOTE: do NOT set customer_creation for subscriptions (causes error)
        allow_promotion_codes=True,
        metadata={
            "promoter_code": ref,
            "attribution_source": "cookie" if ref else "none",
        }
    )
    if request.method == "GET":
        return redirect(session.url, code=302)
    return {"checkout_url": session.url}

@app.get("/success")
def success():
    sid = request.args.get("sid")
    if not sid:
        return "Missing sid", 400

    session = stripe.checkout.Session.retrieve(sid, expand=["subscription", "customer"])

    # Only issue code if checkout fully complete, paid, and subscription exists
    if session.get("status") != "complete" or session.get("payment_status") != "paid" or not session.get("subscription"):
        return "Checkout not finished yet. If you just paid, give it a few seconds and refresh this page.", 400

    customer_id = session["customer"]
    promoter_code = (session.get("metadata") or {}).get("promoter_code", "")

    # Issue one-time link code
    code = secrets.token_urlsafe(6)
    exp = int(time.time()) + CODE_TTL_SECONDS

    tokens = _load_json(TOKENS_FILE)
    tokens[code] = {
        "stripe_customer_id": customer_id,
        "exp": exp,
        "promoter_code": promoter_code
    }
    _save_json(TOKENS_FILE, tokens)

    return f"""
    <html><body style="font-family:sans-serif">
      <h2>Success 🎉</h2>
      <p>Your one-time link code (valid for {CODE_TTL_SECONDS//60} minutes):</p>
      <pre style="font-size:20px">{code}</pre>
      <p>In our Discord, run: <code>/link {code}</code></p>
    </body></html>
    """

@app.get("/cancel")
def cancel():
    return "Canceled."

# --- Dev-only inspectors (guarded) ---
@app.get("/dev/tokens")
def dev_tokens():
    if DEV_DEBUG:
        return _load_json(TOKENS_FILE)
    return ("Not found", 404)

@app.get("/dev/links")
def dev_links():
    if DEV_DEBUG:
        return _load_json(LINKS_FILE)
    return ("Not found", 404)

# --- Discord OAuth2 (Login) ---
def _discord_authorize_url(state):
    from urllib.parse import quote, urlencode
    params = {
        "client_id": DISCORD_CLIENT_ID,
        "redirect_uri": DISCORD_REDIRECT_URI,
        "response_type": "code",
        "scope": "identify",
        "state": state,
        "prompt": "none",
    }
    return "https://discord.com/api/oauth2/authorize?" + urlencode(params, quote_via=quote)

app.secret_key = APP_SECRET


@app.route("/oauth/discord/login")
def discord_login():
    params = {
        "client_id": DISCORD_CLIENT_ID,
        "redirect_uri": DISCORD_REDIRECT_URI,
        "response_type": "code",
        "scope": "identify",
    }
    auth_url = "https://discord.com/api/oauth2/authorize?" + urlencode(params)
    # If running inside the Hugging Face hub iframe, Discord refuses to be embedded.
    # Send the browser to the top-level window.
    html = f"""<!doctype html><html><head><meta charset='utf-8'></head>
<body>
<script>
  (function() {{
    try {{
      if (window.top && window.top !== window.self) {{
        window.top.location = {json.dumps(auth_url)};
      }} else {{
        window.location = {json.dumps(auth_url)};
      }}
    }} catch (e) {{
      window.location = {json.dumps(auth_url)};
    }}
  }})();
</script>
<noscript><a href="{auth_url}">Continue to Discord Login</a></noscript>
</body></html>"""
    return html


@app.route("/oauth/discord/callback")
def discord_callback():
    code = request.args.get("code")
    if not code:
        return "Missing authorization code", 400

    data = {
        "client_id": DISCORD_CLIENT_ID,
        "client_secret": DISCORD_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": DISCORD_REDIRECT_URI,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post("https://discord.com/api/oauth2/token", data=data, headers=headers)
    resp.raise_for_status()
    token_data = resp.json()

    user_resp = requests.get(
        "https://discord.com/api/users/@me",
        headers={"Authorization": f"Bearer {token_data['access_token']}"}
    )
    user = user_resp.json()

    session["discord_user"] = {
        "id": user.get("id"),
        "username": user.get("username"),
        "avatar": user.get("avatar")
    }

    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.pop("discord_user", None)
    return redirect(url_for("index"))

@app.get("/me")
def me():
    return (session.get("discord_user") or {}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")), debug=True)