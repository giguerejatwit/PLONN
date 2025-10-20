import os, json, time, asyncio, logging
import discord
from discord import app_commands
from dotenv import load_dotenv
from aiohttp import web

try:
    import stripe
except Exception:
    stripe = None

load_dotenv()

# -----------------------------
# Config & constants
# -----------------------------
TOKENS_FILE = "tokens.json"
LINKS_FILE = "links.json"

_guild_env = os.getenv("GUILD_ID")
GUILD_ID = int(_guild_env) if _guild_env and _guild_env.isdigit() else None
SUBSCRIBER_ROLE_NAME = os.getenv("SUBSCRIBER_ROLE_NAME", "Gold Member")

# Welcome channel (optional)
_welcome_id = os.getenv("WELCOME_CHANNEL_ID", "").strip()
WELCOME_CHANNEL_ID = int(_welcome_id) if _welcome_id.isdigit() else 0

_general_id = os.getenv("GENERAL_CHANNEL_ID", "").strip()
GENERAL_CHANNEL_ID = int(_general_id) if _general_id.isdigit() else 0

# Stripe Payment Links (no Flask/api needed)
PAYMENT_LINK_GOLD = os.getenv("PAYMENT_LINK_GOLD", "").strip()  # required
def _env_ref_link(code_upper: str) -> str | None:
    return os.getenv(f"REF_LINK_{code_upper}", "").strip() or None

# Stripe webhook (for auto role assignment)
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/stripe/webhook").strip()

# Stripe API (optional) – enables /claim email verification without any web server
STRIPE_SECRET = os.getenv("STRIPE_SECRET", "").strip()
STRIPE_PRICE_ID = os.getenv("PRICE_ID", "").strip()  # the recurring price for Gold Member
if STRIPE_SECRET and stripe:
    stripe.api_key = STRIPE_SECRET
    # Keep Stripe calls from blocking the Discord event loop too long
    try:
        from stripe import http_client as _stripe_http_client
        stripe.default_http_client = _stripe_http_client.RequestsClient(timeout=15)  # seconds
    except Exception:
        pass
    stripe.max_network_retries = 2

# Dynamic Checkout Session config (for API-created sessions)
SUCCESS_URL = os.getenv("SUCCESS_URL", "https://discord.com").strip()
CANCEL_URL = os.getenv("CANCEL_URL", "https://discord.com").strip()

# Known referral codes (normalize case)
KNOWN_REFERRALS = {
    "MINGOSJCE": "MINGOSJCE",  # uses REF_LINK_MINGOSJCE if set
}

# Coupon lookup for referrals
def referral_coupon_id_for(code: str | None) -> str | None:
    """
    Resolve an env var like COUPON_<CODE> to a Stripe coupon id.
    Example: COUPON_MINGOSJCE=coupon_abc123
    """
    if not code:
        return None
    key = f"COUPON_{code.strip().upper()}"
    return os.getenv(key, "").strip() or None

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("plonny")

# -----------------------------
# JSON helpers
# -----------------------------
def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -----------------------------
# Stripe webhook role grant helper
# -----------------------------
async def _grant_gold_role(discord_user_id: int | str, guild_id: int | None):
    """
    Ensure the member has the Gold Member role; create the role if missing.
    """
    try:
        uid = int(discord_user_id)
    except Exception:
        log.warning("Invalid discord_user_id in webhook: %r", discord_user_id)
        return False

    gid = guild_id or GUILD_ID
    if not gid:
        log.warning("No guild_id available for role grant.")
        return False

    guild = bot.get_guild(int(gid))
    if not guild:
        log.warning("Bot not in guild %s or not cached.", gid)
        return False

    role = discord.utils.get(guild.roles, name=SUBSCRIBER_ROLE_NAME)
    if not role:
        try:
            role = await guild.create_role(name=SUBSCRIBER_ROLE_NAME, reason="PLONN subscriber gating (webhook)")
        except discord.Forbidden:
            log.warning("Missing permission to create role in guild %s", gid)
            return False

    member = guild.get_member(uid)
    if member is None:
        try:
            member = await guild.fetch_member(uid)
        except discord.NotFound:
            log.warning("Member %s not found in guild %s", uid, gid)
            return False
        except discord.Forbidden:
            log.warning("Forbidden fetching member %s in guild %s", uid, gid)
            return False

    try:
        if role not in member.roles:
            await member.add_roles(role, reason="PLONN payment verified via Stripe webhook")
    except discord.Forbidden:
        log.warning("No permission to add roles in guild %s", gid)
        return False
    return True

# -----------------------------
# Bot & intents
# -----------------------------
intents = discord.Intents.default()
intents.members = True  # for on_member_join and role ops
intents.guilds = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# -----------------------------
# Utilities
# -----------------------------
def canonical_ref(code: str | None) -> str | None:
    if not code:
        return None
    up = code.strip().upper()
    return KNOWN_REFERRALS.get(up, code.strip().lower())

def find_welcome_channel(guild: discord.Guild) -> discord.TextChannel | None:
    # 1) explicit general channel id
    if GENERAL_CHANNEL_ID:
        ch = guild.get_channel(GENERAL_CHANNEL_ID)
        if isinstance(ch, discord.TextChannel):
            return ch

    # 2) common general channel names
    preferred = {"general", "general-chat", "chat", "lobby"}
    for ch in guild.text_channels:
        if ch.name.lower() in preferred:
            return ch

    # 3) explicit welcome channel id as fallback
    if WELCOME_CHANNEL_ID:
        ch = guild.get_channel(WELCOME_CHANNEL_ID)
        if isinstance(ch, discord.TextChannel):
            return ch

    # 4) system channel
    if guild.system_channel and isinstance(guild.system_channel, discord.TextChannel):
        return guild.system_channel

    # 5) first channel we can speak in
    for ch in guild.text_channels:
        if ch.permissions_for(guild.me).send_messages:
            return ch
    return None

async def send_welcome(guild: discord.Guild, member: discord.Member):
    ch = find_welcome_channel(guild)
    if not ch:
        log.warning("No suitable channel available in %s for welcome message.", guild.name)
        return
    perms = ch.permissions_for(guild.me)
    if not (perms.view_channel and perms.send_messages):
        log.warning("No permission to send in #%s; skipping welcome.", ch.name)
        return

    desc = (
        f"Welcome {member.mention}!\n\n"
        "**I’m Plonny** — here’s what I can do right now:\n"
        "• `/help` — show commands\n"
        "• `/subscribe` — get your Stripe payment link for **Gold Member**\n"
        "• `/referral code:<CODE>` — apply a creator code for a discounted link\n"
        "• `/link code:<one-time>` — link your Discord to your paid subscription\n"
    )
    try:
        await ch.send(desc)
    except discord.Forbidden:
        log.warning("Missing permission to send in %s.", ch)
    except discord.HTTPException:
        log.warning("Failed to send welcome message in %s.", ch)

def build_payment_link(ref_code: str | None = None) -> str:
    """
    Return a Stripe Payment Link.
    - If a referral code is supplied and REF_LINK_<CODE> is defined in env, use that link.
    - Else fall back to PAYMENT_LINK_GOLD.
    """
    if ref_code:
        up = canonical_ref(ref_code)
        if up:
            env_link = _env_ref_link(up.upper())
            if env_link:
                return env_link
    if not PAYMENT_LINK_GOLD:
        raise RuntimeError("PAYMENT_LINK_GOLD is not configured.")
    return PAYMENT_LINK_GOLD

# ------------------------------------------
# Stripe email subscription verification
# ------------------------------------------
async def has_active_gold_by_email(email: str) -> bool:
    """
    Check via Stripe API whether the given email has an active subscription
    to the configured STRIPE_PRICE_ID. Requires STRIPE_SECRET & stripe lib.
    """
    if not stripe or not STRIPE_SECRET or not STRIPE_PRICE_ID:
        raise RuntimeError("Stripe verification not configured: install 'stripe' and set STRIPE_SECRET and PRICE_ID.")

    # 1) Find customers by email (most recent first)
    customers = stripe.Customer.search(query=f"email:'{email}'", limit=5)
    # If search API not available, fallback to list (older accounts)
    if not hasattr(customers, 'data'):
        customers = stripe.Customer.list(email=email, limit=5)

    for cust in customers.data:
        # 2) Find subscriptions for this customer
        subs = stripe.Subscription.list(customer=cust.id, status="all", expand=["data.items.price"])
        for sub in subs.auto_paging_iter():
            if sub.status in ("active", "trialing"):
                for item in sub.items.data:
                    if getattr(item.price, "id", None) == STRIPE_PRICE_ID:
                        return True
    return False

# -----------------------------
# Dynamic Stripe Checkout Session (for /subscribe and /referral)
# -----------------------------
def _create_checkout_session_for_user(
    user: discord.User | discord.Member,
    guild_id: int | None,
    ref_code: str | None = None,
) -> str:
    """
    Returns a Stripe Checkout URL for a subscription to STRIPE_PRICE_ID with metadata:
    - discord_user_id
    - guild_id
    - ref_code (if provided)
    If a COUPON_<CODE> env exists matching ref_code, it will be applied.
    """
    if not (stripe and STRIPE_SECRET and STRIPE_PRICE_ID):
        raise RuntimeError("Stripe not configured. Set STRIPE_SECRET and PRICE_ID, and install 'stripe'.")

    discounts = []
    coup = referral_coupon_id_for(ref_code)
    if coup:
        discounts = [{"coupon": coup}]
    meta = {"discord_user_id": str(user.id)}
    if guild_id:
        meta["guild_id"] = str(guild_id)
    if ref_code:
        meta["ref_code"] = str(ref_code)

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=SUCCESS_URL,
        cancel_url=CANCEL_URL,
        metadata=meta,
        client_reference_id=str(user.id),
        allow_promotion_codes=True,
        discounts=discounts or None,
    )
    return session.url

# Async wrapper to run the blocking Stripe call in a worker thread
async def create_checkout_session_for_user_async(
    user: discord.User | discord.Member,
    guild_id: int | None,
    ref_code: str | None = None,
) -> str:
    # Run blocking Stripe HTTP in a worker thread so we don't block the event loop/heartbeats
    return await asyncio.to_thread(_create_checkout_session_for_user, user, guild_id, ref_code)

# -----------------------------
# Stripe Webhook (aiohttp)
# -----------------------------
async def stripe_webhook_handler(request: web.Request):
    if not (stripe and STRIPE_WEBHOOK_SECRET):
        return web.Response(status=400, text="Stripe webhook unconfigured")
    try:
        payload = await request.read()
        sig = request.headers.get("Stripe-Signature", "")
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        log.exception("Webhook signature verification failed: %s", e)
        return web.Response(status=400, text="Bad signature")

    etype = event.get("type")
    data = event.get("data", {}).get("object", {})

    # Prefer checkout.session.completed to capture new subs; fall back to invoice.paid
    if etype == "checkout.session.completed":
        meta = data.get("metadata", {}) or {}
        discord_id = meta.get("discord_user_id")
        gid = meta.get("guild_id")
        ok = await _grant_gold_role(discord_id, int(gid) if gid else GUILD_ID)
        log.info("checkout.session.completed handled for discord_id=%s ok=%s", discord_id, ok)
    elif etype == "invoice.paid":
        # Recurring invoice paid — try to resolve discord id from subscription/customer metadata if present
        subscription_id = data.get("subscription")
        try:
            sub = stripe.Subscription.retrieve(subscription_id, expand=["metadata", "customer", "items.data.price"])
            meta = getattr(sub, "metadata", {}) or {}
            discord_id = meta.get("discord_user_id")
            gid = meta.get("guild_id")
            if discord_id:
                ok = await _grant_gold_role(discord_id, int(gid) if gid else GUILD_ID)
                log.info("invoice.paid handled for discord_id=%s ok=%s", discord_id, ok)
        except Exception as e:
            log.exception("invoice.paid enrichment failed: %s", e)

    return web.Response(status=200, text="ok")

_web_runner: web.AppRunner | None = None

async def start_webhook_server():
    global _web_runner
    app = web.Application()
    app.router.add_post(WEBHOOK_PATH, stripe_webhook_handler)
    _web_runner = web.AppRunner(app)
    await _web_runner.setup()
    site = web.TCPSite(_web_runner, "0.0.0.0", WEBHOOK_PORT)
    await site.start()
    log.info("Stripe webhook listening on 0.0.0.0:%s%s", WEBHOOK_PORT, WEBHOOK_PATH)

# -----------------------------
# Events
# -----------------------------
@bot.event
async def on_ready():
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    try:
        if GUILD_ID:
            guild = discord.Object(id=GUILD_ID)
            # Make all global commands available in this guild instantly
            tree.copy_global_to(guild=guild)
            synced = await tree.sync(guild=guild)
            log.info("Slash commands copied & synced for guild %s (count=%d).", GUILD_ID, len(synced))
        else:
            synced = await tree.sync()
            log.info("Slash commands synced globally (count=%d).", len(synced))
    except Exception as e:
        log.exception("Slash sync failed: %s", e)

    # Start webhook server (only once)
    if STRIPE_WEBHOOK_SECRET:
        # Avoid double-start on reconnects by checking a custom attribute
        if not getattr(bot, "_webhook_started", False):
            asyncio.create_task(start_webhook_server())
            bot._webhook_started = True
    for guild in bot.guilds:
        print(f"[DEBUG] Commands visible in {guild.name}:")
        cmds = await tree.fetch_commands(guild=guild)
        for c in cmds:
            print(f" - {c.name}")

@bot.event
async def on_member_join(member: discord.Member):
    await send_welcome(member.guild, member)

# -----------------------------
# Slash commands
# -----------------------------
@tree.command(name="help", description="Show what Plonny can do.")
async def help_cmd(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Plonny — Help",
        description=(
            "**/help** — show this help\n"
            "**/subscribe** — create your personal Stripe checkout (auto-embeds your Discord ID)\n"
            "**/referral `code`** — same as subscribe, with coupon if configured\n"
            "**/myid** — show your numeric Discord ID (for support)\n"
            "**/link `code`** — (optional) legacy one-time code flow"
        ),
        color=0x5865F2
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)

@tree.command(name="subscribe", description="Get your Stripe checkout link for Gold Member.")
async def subscribe_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        url = await create_checkout_session_for_user_async(interaction.user, interaction.guild_id or GUILD_ID, ref_code=None)
        await interaction.followup.send(
            f"Here’s your Stripe checkout for **Gold Member**:\n{url}\n"
            "After payment, your Discord will be auto-linked via metadata (or use `/claim email`).",
            ephemeral=True
        )
    except Exception as e:
        log.exception("subscribe error: %s", e)
        await interaction.followup.send(f"Sorry — I couldn’t create a checkout: {e}", ephemeral=True)


@tree.command(name="referral", description="Use a creator referral code for a discounted subscription link.")
@app_commands.describe(code="Referral code")
async def referral_cmd(interaction: discord.Interaction, code: str):
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        # known code normalization retained
        url = await create_checkout_session_for_user_async(interaction.user, interaction.guild_id or GUILD_ID, ref_code=code)
        note = ""
        if referral_coupon_id_for(code):
            note = " (coupon applied)"
        await interaction.followup.send(
            f"Referral `{code}` applied{note}. Here’s your checkout:\n{url}",
            ephemeral=True
        )
    except Exception as e:
        log.exception("referral error: %s", e)
        await interaction.followup.send(f"Sorry — I couldn’t create a referral checkout: {e}", ephemeral=True)
@tree.command(name="myid", description="Show your Discord numeric user ID (ephemeral)")
async def myid_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(f"Your Discord ID is `{interaction.user.id}`.", ephemeral=True)

# -----------------------------
# /claim: Gold via email verification (Stripe API)
# -----------------------------


@tree.command(name="sync", description="Admin: force re-sync slash commands in this server")
@app_commands.checks.has_permissions(administrator=True)
async def sync_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        if interaction.guild:
            synced = await tree.sync(guild=interaction.guild)
            await interaction.followup.send(f"✅ Synced {len(synced)} commands for **{interaction.guild.name}**.", ephemeral=True)
        else:
            synced = await tree.sync()
            await interaction.followup.send(f"✅ Synced {len(synced)} commands globally.", ephemeral=True)
    except Exception as e:
        log.exception("manual sync error: %s", e)
        await interaction.followup.send(f"❌ Sync failed: {e}", ephemeral=True)

@tree.command(name="ping", description="Check if Plonny is responding (ephemeral)")
async def ping_cmd(interaction: discord.Interaction):
    await interaction.response.send_message("🏓 Pong! Plonny is alive.", ephemeral=True)

# -----------------------------
# Admin: manual Gold grant command
# -----------------------------
@tree.command(name="grant", description="Admin: grant Gold Member role to a user (manual)")
@app_commands.checks.has_permissions(administrator=True)
@app_commands.describe(user="User to grant Gold Member")
async def grant_cmd(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    guild = interaction.guild
    if not guild:
        return await interaction.followup.send("Use this in a server.", ephemeral=True)
    ok = await _grant_gold_role(user.id, guild.id)
    if ok:
        await interaction.followup.send(f"✅ Granted Gold Member to {user.mention}.", ephemeral=True)
    else:
        await interaction.followup.send("❌ Could not grant role. Check bot permissions & role order.", ephemeral=True)

# -----------------------------
# Existing /link command (unchanged logic)
# -----------------------------
if GUILD_ID:
    @tree.command(name="link", description="Link your Discord to your PLONN subscription", guild=discord.Object(id=GUILD_ID))
    @app_commands.describe(code="One-time code from the PLONN success page")
    async def link_cmd(interaction: discord.Interaction, code: str):
        await interaction.response.defer(ephemeral=True)

        tokens = _load_json(TOKENS_FILE)
        token = tokens.get(code)
        now = int(time.time())

        if not token:
            return await interaction.followup.send("❌ Invalid code. Double-check and try again.", ephemeral=True)
        if token["exp"] < now:
            return await interaction.followup.send("⌛ Code expired. Please generate a new one from the success page.", ephemeral=True)

        stripe_customer_id = token["stripe_customer_id"]

        # Consume the token
        del tokens[code]
        _save_json(TOKENS_FILE, tokens)

        # Store the link
        links = _load_json(LINKS_FILE)
        links[str(interaction.user.id)] = stripe_customer_id
        _save_json(LINKS_FILE, links)

        # Assign Subscriber role
        guild = interaction.guild
        if not guild:
            return await interaction.followup.send("This command must be used in the server.", ephemeral=True)

        role = discord.utils.get(guild.roles, name=SUBSCRIBER_ROLE_NAME)
        if not role:
            try:
                role = await guild.create_role(name=SUBSCRIBER_ROLE_NAME, reason="PLONN subscriber gating")
            except discord.Forbidden:
                return await interaction.followup.send("I need permission to create/assign roles. Ask an admin.", ephemeral=True)

        member = guild.get_member(interaction.user.id) or await guild.fetch_member(interaction.user.id)
        try:
            if role not in member.roles:
                await member.add_roles(role, reason="PLONN subscriber verified")
        except discord.Forbidden:
            return await interaction.followup.send("I don't have permission to add roles. Move my role above Subscriber.", ephemeral=True)

        await interaction.followup.send("✅ Linked! You now have Subscriber access.", ephemeral=True)
else:
    @tree.command(name="link", description="Link your Discord to your PLONN subscription")
    @app_commands.describe(code="One-time code from the PLONN success page")
    async def link_cmd(interaction: discord.Interaction, code: str):
        await interaction.response.defer(ephemeral=True)

        tokens = _load_json(TOKENS_FILE)
        token = tokens.get(code)
        now = int(time.time())

        if not token:
            return await interaction.followup.send("❌ Invalid code. Double-check and try again.", ephemeral=True)
        if token["exp"] < now:
            return await interaction.followup.send("⌛ Code expired. Please generate a new one from the success page.", ephemeral=True)

        stripe_customer_id = token["stripe_customer_id"]

        # Consume the token
        del tokens[code]
        _save_json(TOKENS_FILE, tokens)

        # Store the link
        links = _load_json(LINKS_FILE)
        links[str(interaction.user.id)] = stripe_customer_id
        _save_json(LINKS_FILE, links)

        # Assign Subscriber role
        guild = interaction.guild
        if not guild:
            return await interaction.followup.send("This command must be used in the server.", ephemeral=True)

        role = discord.utils.get(guild.roles, name=SUBSCRIBER_ROLE_NAME)
        if not role:
            try:
                role = await guild.create_role(name=SUBSCRIBER_ROLE_NAME, reason="PLONN subscriber gating")
            except discord.Forbidden:
                return await interaction.followup.send("I need permission to create/assign roles. Ask an admin.", ephemeral=True)

        member = guild.get_member(interaction.user.id) or await guild.fetch_member(interaction.user.id)
        try:
            if role not in member.roles:
                await member.add_roles(role, reason="PLONN subscriber verified")
        except discord.Forbidden:
            return await interaction.followup.send("I don't have permission to add roles. Move my role above Subscriber.", ephemeral=True)

        await interaction.followup.send("✅ Linked! You now have Subscriber access.", ephemeral=True)

# -----------------------------
# Run
# -----------------------------
bot.run(os.getenv("DISCORD_BOT_TOKEN"))