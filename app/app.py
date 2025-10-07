import os, json, time, secrets
from flask import Flask, request, make_response, redirect
from dotenv import load_dotenv
import stripe

load_dotenv()

# --- Config from .env ---
STRIPE_SECRET = os.getenv("STRIPE_SECRET")
PRICE_ID = os.getenv("PRICE_ID")
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")
CODE_TTL_SECONDS = int(os.getenv("CODE_TTL_SECONDS", "900"))  # 15 minutes by default
DEV_DEBUG = os.getenv("DEV_DEBUG") == "1"

if not STRIPE_SECRET:
  raise RuntimeError("Missing STRIPE_SECRET in .env")
if not PRICE_ID:
  raise RuntimeError("Missing PRICE_ID in .env")

stripe.api_key = STRIPE_SECRET

TOKENS_FILE = "tokens.json"   # { code: { "stripe_customer_id": "...", "exp": 123456, "promoter_code": "..." } }
LINKS_FILE  = "links.json"    # { discord_user_id: "stripe_customer_id" }

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)