import os, json, time, asyncio
import discord
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

TOKENS_FILE = "tokens.json"
LINKS_FILE = "links.json"

_guild_env = os.getenv("GUILD_ID")
GUILD_ID = int(_guild_env) if _guild_env and _guild_env.isdigit() else None
SUBSCRIBER_ROLE_NAME = os.getenv("SUBSCRIBER_ROLE_NAME", "Subscriber")

def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

intents = discord.Intents.default()
intents.members = True
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    if GUILD_ID:
        guild = discord.Object(id=GUILD_ID)
        await tree.sync(guild=guild)
        print(f"Slash commands synced for guild {GUILD_ID}.")
    else:
        await tree.sync()
        print("Slash commands synced globally (no GUILD_ID provided).")

# Define the /link command with or without a guild scope based on env
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

bot.run(os.getenv("DISCORD_BOT_TOKEN"))