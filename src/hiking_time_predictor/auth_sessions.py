"""Encrypted server-side Supabase sessions, shared by Streamlit and FastAPI.

SQLite is only an authentication credential vault, not an application database.
The two processes must share its private persistent volume and encryption key.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
import secrets
import sqlite3
import time
from urllib.parse import urlparse
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from supabase import ClientOptions, create_client
from supabase_auth import SyncMemoryStorage


class AuthenticationError(ValueError):
    pass


@dataclass(frozen=True)
class UserSession:
    user_id: str
    access_token: str = field(repr=False)
    user: dict = field(repr=False)
    created_at: float


class SessionVault:
    def __init__(self, path, key):
        self.path = path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        self.cipher = Fernet(key.encode())
        # Create with restrictive permissions before opening SQLite.
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS vault (key TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                       "user_id TEXT, expires REAL NOT NULL, payload BLOB NOT NULL)")

    @contextmanager
    def transaction(self):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db

    @staticmethod
    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def put(self, db, key, kind, value, ttl, user_id=None):
        db.execute("INSERT OR REPLACE INTO vault VALUES (?, ?, ?, ?, ?)",
                   (self.digest(key), kind, user_id, time.time() + ttl,
                    self.cipher.encrypt(json.dumps(value).encode())))

    def get(self, db, key, kind, consume=False):
        row = db.execute("SELECT expires,payload FROM vault WHERE key=? AND kind=?",
                         (self.digest(key), kind)).fetchone()
        if consume:
            db.execute("DELETE FROM vault WHERE key=?", (self.digest(key),))
        if row is None or row[0] <= time.time():
            raise AuthenticationError("Your session has expired. Continue with Google again.")
        try:
            return json.loads(self.cipher.decrypt(row[1]))
        except (InvalidToken, ValueError):
            raise AuthenticationError("Session unavailable. Continue with Google again.") from None

    def delete(self, key):
        with self.transaction() as db:
            db.execute("DELETE FROM vault WHERE key=?", (self.digest(key),))

    def delete_user(self, user_id):
        with self.transaction() as db:
            db.execute("DELETE FROM vault WHERE user_id=?", (user_id,))


class AuthManager:
    FLOW_TTL = 600
    SESSION_TTL = 7 * 24 * 3600

    def __init__(self, config, *, vault=None, client_factory=None):
        self.config = config
        self.vault = vault or SessionVault(config.session_db, config.session_key)
        self.client_factory = client_factory

    def client(self, storage=None):
        if self.client_factory:
            return self.client_factory(storage)
        return create_client(self.config.supabase_url, self.config.publishable_key,
                             options=ClientOptions(flow_type="pkce", auto_refresh_token=False,
                                                   persist_session=False, storage=storage or SyncMemoryStorage(),
                                                   postgrest_client_timeout=30))

    def begin(self):
        state = secrets.token_urlsafe(32)
        storage = SyncMemoryStorage()
        client = self.client(storage)
        response = client.auth.sign_in_with_oauth({"provider": "google", "options": {
            "redirect_to": self.config.auth_url + "/auth/callback?state=" + state,
            "query_params": {"prompt": "select_account"}}})
        url = response.url
        expected = urlparse(self.config.supabase_url)
        actual = urlparse(url)
        if (actual.scheme, actual.netloc, actual.path) != (expected.scheme, expected.netloc,
                                                             expected.path + "/auth/v1/authorize"):
            raise AuthenticationError("Unexpected OAuth authorization URL.")
        with self.vault.transaction() as db:
            db.execute("DELETE FROM vault WHERE expires < ?", (time.time(),))
            self.vault.put(db, state, "flow", {"storage": storage.storage}, self.FLOW_TTL)
        return state, url

    def finish(self, state, cookie_state, code):
        if (not state or not cookie_state or len(state) > 128 or not code or len(code) > 4096
                or not secrets.compare_digest(state, cookie_state)):
            raise AuthenticationError("Login could not be verified. Start sign-in again.")
        # Commit consumption before the network exchange: failures cannot replay a flow.
        with self.vault.transaction() as db:
            flow = self.vault.get(db, state, "flow", consume=True)
        storage = SyncMemoryStorage()
        storage.storage.update(flow["storage"])
        try:
            response = self.client(storage).auth.exchange_code_for_session({"auth_code": code})
            session = response.session
            verified = self.client().auth.get_user(session.access_token).user
            if "google" not in (verified.app_metadata.get("providers") or [verified.app_metadata.get("provider")]):
                raise AuthenticationError("A Google account is required.")
            user_id = str(UUID(str(verified.id)))
            value = {"user_id": user_id, "access_token": session.access_token, "refresh_token": session.refresh_token,
                     "expires_at": session.expires_at, "created_at": time.time()}
        except Exception:
            raise AuthenticationError("Google sign-in failed. Start sign-in again.") from None
        sid = secrets.token_urlsafe(48)
        with self.vault.transaction() as db:
            self.vault.put(db, sid, "session", value, self.SESSION_TTL, user_id)
        return sid

    def restore(self, sid):
        if not sid or len(sid) > 128:
            raise AuthenticationError("Continue with Google to access your hiking history.")
        try:
            # Serialize refresh-token rotation across both processes and browser tabs.
            with self.vault.transaction() as db:
                value = self.vault.get(db, sid, "session")
                client = self.client()
                if value["expires_at"] <= time.time() + 60:
                    session = client.auth.refresh_session(value["refresh_token"]).session
                    value.update(access_token=session.access_token, refresh_token=session.refresh_token,
                                 expires_at=session.expires_at)
                user = client.auth.get_user(value["access_token"]).user
                user_id = str(UUID(str(user.id)))
                if user_id != value['user_id']:
                    raise AuthenticationError('Session identity changed.')
                remaining = self.SESSION_TTL - (time.time() - value["created_at"])
                if remaining <= 0:
                    raise AuthenticationError("Session expired.")
                self.vault.put(db, sid, "session", value, remaining, user_id)
            return UserSession(user_id, value["access_token"], user.model_dump(mode="json"), value["created_at"])
        except Exception:
            self.vault.delete(sid)
            raise AuthenticationError("Session expired or revoked. Continue with Google again.") from None

    def logout(self, sid):
        user_id = None
        try:
            session = self.restore(sid)
            user_id = session.user_id
            # This SDK endpoint uses the user's JWT, not a service-role key.
            self.client().auth.admin.sign_out(session.access_token, "global")
        except Exception:
            pass
        finally:
            self.vault.delete(sid)
            if user_id:
                self.vault.delete_user(user_id)
