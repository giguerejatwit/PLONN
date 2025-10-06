# Use a small Python image
FROM python:3.11-slim

# Work under /app
WORKDIR /app

# Copy your whole repo
COPY . /app

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# HF Spaces set $PORT (typically 7860)
ENV PORT=7860
EXPOSE 7860

# Your Flask app lives in discord/app.py with Flask instance named "app"
WORKDIR /app/discord
CMD ["gunicorn", "app:app", "--workers", "2", "--threads", "4", "--timeout", "60", "--bind", "0.0.0.0:7860"]