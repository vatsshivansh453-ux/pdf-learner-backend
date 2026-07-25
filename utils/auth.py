"""
OAuth2 (Google + GitHub) authentication for PDF-LEARNER.

Flow:
  1. Frontend redirects the browser to /auth/login/{provider}
  2. We redirect to the provider's consent screen (Authlib handles this)
  3. Provider redirects back to /auth/callback/{provider}
  4. We fetch the user's profile, upsert a local user row,
     issue a JWT, and set it as an httpOnly cookie
  5. We redirect the browser back to the frontend
  6. The frontend calls GET /auth/me (cookie sent automatically) to
     find out who's logged in
"""

import os
from datetime import datetime, timedelta, timezone

import httpx
import jwt
from authlib.integrations.starlette_client import OAuth
from fastapi import Cookie, HTTPException, status
from dotenv import load_dotenv

from utils.memory import get_or_create_user, get_user_by_id

load_dotenv()

# =====================================================
# CONFIG
# =====================================================

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

COOKIE_NAME = "pdf_learner_token"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# =====================================================
# OAUTH CLIENT REGISTRY
# =====================================================

oauth = OAuth()

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(
        name="github",
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "read:user user:email"},
    )


# =====================================================
# JWT HELPERS
# =====================================================

def create_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None


# =====================================================
# PROVIDER PROFILE NORMALIZATION
# =====================================================

async def upsert_user_from_google(userinfo: dict):
    return get_or_create_user(
        provider="google",
        provider_id=userinfo["sub"],
        email=userinfo.get("email"),
        name=userinfo.get("name") or userinfo.get("email", "Google User"),
        avatar_url=userinfo.get("picture"),
    )


async def upsert_user_from_github(access_token: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient() as client:
        profile_resp = await client.get("https://api.github.com/user", headers=headers)
        profile_resp.raise_for_status()
        profile = profile_resp.json()

        email = profile.get("email")

        if not email:
            emails_resp = await client.get("https://api.github.com/user/emails", headers=headers)
            if emails_resp.status_code == 200:
                emails = emails_resp.json()
                primary = next((e for e in emails if e.get("primary")), None)
                email = (primary or (emails[0] if emails else {})).get("email")

    return get_or_create_user(
        provider="github",
        provider_id=str(profile["id"]),
        email=email,
        name=profile.get("name") or profile.get("login"),
        avatar_url=profile.get("avatar_url"),
    )


# =====================================================
# CURRENT USER DEPENDENCY
# =====================================================

def get_current_user(pdf_learner_token: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    """
    FastAPI dependency. Reads the JWT from the httpOnly cookie,
    validates it, and returns the user record. Raises 401 if missing/invalid.
    """

    if not pdf_learner_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user_id = decode_access_token(pdf_learner_token)

    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")

    user = get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


def get_optional_user(pdf_learner_token: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    """Same as get_current_user but returns None instead of raising."""

    if not pdf_learner_token:
        return None

    user_id = decode_access_token(pdf_learner_token)

    if not user_id:
        return None

    return get_user_by_id(user_id)
