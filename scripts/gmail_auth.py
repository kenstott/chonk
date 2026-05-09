#!/usr/bin/env python3
"""One-time Gmail OAuth2 consent flow. Run once to save token to ~/.chonk/gmail_token.json.

Usage:
    python scripts/gmail_auth.py

1. A URL is printed — open it in a browser and authorize.
2. Google redirects to localhost:8000 (which will show a connection error — that's fine).
3. Copy the full URL from the browser address bar and paste it here.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).parents[1] / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE)
    except ImportError:
        pass

from google_auth_oauthlib.flow import InstalledAppFlow

_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
_TOKEN_PATH = Path.home() / ".chonk" / "gmail_token.json"
_REDIRECT_URI = "http://localhost:8000/"

client_id = os.environ["GOOGLE_EMAIL_CLIENT_ID"]
client_secret = os.environ["GOOGLE_EMAIL_CLIENT_SECRET"]

client_config = {
    "installed": {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uris": [_REDIRECT_URI],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

flow = InstalledAppFlow.from_client_config(client_config, _SCOPES)
flow.redirect_uri = _REDIRECT_URI

auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
print("\nOpen this URL in your browser:")
print(auth_url)
print("\nAfter authorizing, you'll be redirected to localhost:8000 (may show 'connection refused').")
print("Copy the FULL URL from the browser address bar and paste it here:")

redirect_response = input("> ").strip()
flow.fetch_token(authorization_response=redirect_response)

_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
_TOKEN_PATH.write_text(flow.credentials.to_json())
print(f"\nToken saved to {_TOKEN_PATH}")
