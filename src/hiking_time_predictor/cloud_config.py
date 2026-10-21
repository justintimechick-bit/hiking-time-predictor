"""Server configuration. No credentials are needed to import or test the ML code."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class CloudConfig:
    supabase_url: str
    publishable_key: str = field(repr=False)
    app_url: str = "http://localhost:8501"
    auth_url: str = "http://localhost:8000"
    session_db: Path = Path(".auth/sessions.sqlite3")
    session_key: str = field(default="", repr=False)
    model_key: str = field(default="", repr=False)
    admin_key: str = field(default="", repr=False)
    secure_cookies: bool = False
    auth_mode: str = "service"
    google_client_id: str = ""

    @classmethod
    def from_env(cls, *, include_admin=False):
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / '.env', override=False)
        def setting(name, default=""):
            if name in os.environ:
                return os.environ[name]
            # Community Cloud supplies secrets through Streamlit, not .env files.
            import streamlit as st
            try:
                return str(st.secrets.get(name, default))
            except FileNotFoundError:
                return default

        url = setting("SUPABASE_URL").rstrip("/")
        key = setting("SUPABASE_PUBLISHABLE_KEY")
        if not url or not key:
            raise ValueError("Configure SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY in .env or Streamlit secrets. See docs/STREAMLIT_CLOUD.md.")
        result = cls(url, key, setting("APP_URL", "http://localhost:8501").rstrip("/"),
                     setting("AUTH_PUBLIC_URL", "http://localhost:8000").rstrip("/"),
                     Path(setting("AUTH_SESSION_DB", ".auth/sessions.sqlite3")).resolve(),
                     setting("AUTH_SESSION_KEY"), setting("MODEL_SIGNING_KEY"),
                     setting("SUPABASE_SECRET_KEY") if include_admin else "",
                     setting("COOKIE_SECURE", "true").lower() == "true",
                     setting("AUTH_MODE", "service"), setting("GOOGLE_CLIENT_ID"))
        result.validate()
        return result

    def validate(self):
        from cryptography.fernet import Fernet
        if self.auth_mode not in ("service", "streamlit"):
            raise ValueError("AUTH_MODE must be service or streamlit.")
        if self.auth_mode == "service":
            try:
                Fernet(self.session_key.encode())
            except Exception:
                raise ValueError("AUTH_SESSION_KEY must be a Fernet key; follow the setup guide.") from None
        elif not self.google_client_id.endswith(".apps.googleusercontent.com"):
            raise ValueError("Configure GOOGLE_CLIENT_ID for Streamlit sign-in. See docs/STREAMLIT_CLOUD.md.")
        if len(self.model_key.encode()) < 32:
            raise ValueError("MODEL_SIGNING_KEY needs at least 32 random bytes of text.")
        urls = (self.supabase_url, self.app_url) if self.auth_mode == "streamlit" else (self.supabase_url, self.app_url, self.auth_url)
        for url in urls:
            parsed = urlparse(url)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('','/')):
                raise ValueError("Use explicit HTTP(S) origins without credentials, queries or fragments.")
            if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1"):
                raise ValueError("Non-local services require HTTPS.")
        if self.auth_mode == "service" and urlparse(self.app_url).hostname != urlparse(self.auth_url).hostname:
            raise ValueError("APP_URL and AUTH_PUBLIC_URL must share a hostname for the session cookie.")
        if self.auth_mode == "service" and not self.secure_cookies and any(urlparse(u).hostname not in ("localhost", "127.0.0.1")
                                           for u in (self.app_url, self.auth_url)):
            raise ValueError("COOKIE_SECURE=false is allowed only on loopback.")
        if self.auth_mode == "service" and self.secure_cookies and any(urlparse(u).scheme != "https" for u in (self.app_url, self.auth_url)):
            raise ValueError("Secure cookies require HTTPS. For localhost only, set COOKIE_SECURE=false.")
        # A legacy service_role JWT or new secret key must never be the end-user key.
        if self.publishable_key.startswith("sb_secret_"):
            raise ValueError("Use a publishable key for SUPABASE_PUBLISHABLE_KEY.")
        if self.publishable_key.startswith("eyJ"):
            import base64, json
            try:
                payload = json.loads(base64.urlsafe_b64decode(self.publishable_key.split(".")[1] + "=="))
            except Exception:
                raise ValueError("Invalid legacy anonymous key.") from None
            if payload.get("role") != "anon":
                raise ValueError("The application key must have the anon role.")

    @property
    def session_cookie(self):
        return "__Host-hiking_session" if self.secure_cookies else "hiking_session_dev"

    @property
    def flow_cookie(self):
        return "__Host-hiking_flow" if self.secure_cookies else "hiking_flow_dev"
