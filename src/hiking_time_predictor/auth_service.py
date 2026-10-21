"""Focused OAuth callback/cookie service. Run with python -m ...auth_service."""
import hashlib
import hmac
import html
import secrets
import time
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from .auth_sessions import AuthManager, AuthenticationError
from .cloud_config import CloudConfig


def create_app(config=None, *, manager=None, cleanup=None):
    config = config or CloudConfig.from_env(include_admin=True)
    manager = manager or AuthManager(config)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def cookie(response, name, value, ttl):
        response.set_cookie(name, value, max_age=ttl, secure=config.secure_cookies,
                            httponly=True, samesite='lax', path='/')

    def clear(response, name):
        response.delete_cookie(name, path='/', secure=config.secure_cookies, httponly=True, samesite='lax')

    @app.middleware('http')
    async def private_response(request, call_next):
        response = await call_next(request)
        response.headers.update({'Cache-Control': 'no-store', 'Pragma': 'no-cache',
            # no-referrer makes browsers send Origin: null on HTML form POSTs.
            # Send the origin only, preserving CSRF checks without leaking OAuth URLs.
            'Referrer-Policy': 'strict-origin', 'X-Content-Type-Options': 'nosniff',
            # Some browsers apply form-action to the POST's redirect too.
            # Streamlit has a separate port locally, so 'self' alone blocks
            # returning to the app after an otherwise successful logout.
            'Content-Security-Policy': (
                "default-src 'none'; style-src 'unsafe-inline'; "
                f"form-action 'self' {config.app_url}; frame-ancestors 'none'")})
        return response

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/auth/login')
    def login():
        try:
            state, url = manager.begin()
        except Exception:
            return HTMLResponse('Sign-in is temporarily unavailable. Check server configuration.', status_code=503)
        response = RedirectResponse(url, status_code=303)
        cookie(response, config.flow_cookie, state, manager.FLOW_TTL)
        return response

    @app.get('/auth/callback')
    def callback(request: Request, state: str = '', code: str = ''):
        try:
            sid = manager.finish(state, request.cookies.get(config.flow_cookie), code)
        except AuthenticationError:
            response = HTMLResponse('Sign-in failed or expired. Return to the app and select Continue with Google again.', status_code=400)
        else:
            old = request.cookies.get(config.session_cookie)
            if old:
                manager.vault.delete(old)
            response = RedirectResponse(config.app_url, status_code=303)
            cookie(response, config.session_cookie, sid, manager.SESSION_TTL)
        clear(response, config.flow_cookie)
        return response

    def csrf(sid, action):
        return hmac.new(config.session_key.encode(), (sid + '|' + action).encode(), hashlib.sha256).hexdigest()

    actions = {'logout': ('Are you sure you want to log out?', None),
               'clear': ('Permanently delete all hiking data, GPX files, models and predictions. Your Google sign-in account remains.', 'DELETE MY HIKING DATA'),
               'delete': ('Permanently delete all hiking data, GPX files, models, predictions, profile and Supabase sign-in identity. Your Google account itself is not deleted.', 'DELETE MY ACCOUNT')}

    @app.get('/account/{action}')
    def confirmation(action: str, request: Request):
        if action not in actions:
            return HTMLResponse('Not found.', status_code=404)
        sid = request.cookies.get(config.session_cookie, '')
        try:
            manager.restore(sid)
        except AuthenticationError:
            return RedirectResponse(config.app_url, status_code=303)
        explanation, phrase = actions[action]
        token = csrf(sid, action)
        if action == 'logout':
            heading = 'Log out'
            # Separate forms keep the choice in FormData even when a browser or
            # extension submits without including the clicked submitter button.
            controls = ''.join(
                f'<form method="post" action="/account/logout" class="logout-choice">'
                f'<input type="hidden" name="csrf" value="{token}">'
                f'<input type="hidden" name="confirmation" value="{choice}">'
                f'<button type="submit" class="{style}">{label}</button></form>'
                for choice, label, style in (('yes', 'Yes, log out', ''),
                                              ('no', 'No, stay signed in', 'secondary')))
            instruction = ''
        else:
            heading = 'Account confirmation'
            instruction = f'<p>Type <strong>{phrase}</strong> to continue.</p>'
            controls = '''<input name="confirmation" autocomplete="off" required maxlength="40">
            <button type="submit">Confirm</button>'''
            controls = (f'<form method="post" action="/account/{action}">'
                        f'<input type="hidden" name="csrf" value="{token}">{controls}</form>')
        page = f'''<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Hiking account</title><style>
        body {{ font:18px system-ui;max-width:650px;margin:60px auto;padding:24px;background:#f8faf8;color:#243b32; }}
        h1 {{ color:#214936; }}
        button {{ font:inherit;padding:12px 18px;margin:8px 8px 8px 0;border:1px solid #24533f;
            border-radius:6px;background:#24533f;color:white;cursor:pointer; }}
        .secondary {{ background:white;color:#24533f; }}
        .logout-choice {{ display:inline-block; }}
        input {{ font:inherit;padding:10px;max-width:100%;box-sizing:border-box; }}
        a {{ color:#24533f; }}
        </style></head><body>
        <h1>{heading}</h1><p>{explanation}</p>{instruction}
        {controls}<p><a href="{html.escape(config.app_url)}">Return to app</a></p></body></html>'''
        return HTMLResponse(page)

    @app.post('/account/{action}')
    async def confirmed_action(action: str, request: Request):
        if action not in actions:
            return HTMLResponse('Not found.', status_code=404)
        # SameSite cookies plus an Origin check plus an action-bound CSRF token.
        if request.headers.get('origin') != config.auth_url:
            return HTMLResponse('Request origin rejected.', status_code=403)
        body = b''
        async for chunk in request.stream():
            body += chunk
            if len(body) > 2048:
                return HTMLResponse('Request too large.', status_code=413)
        data = parse_qs(body.decode('utf-8', errors='replace'))
        sid = request.cookies.get(config.session_cookie, '')
        choice = data.get('confirmation', [''])[0]
        valid_csrf = secrets.compare_digest(data.get('csrf', [''])[0].encode(), csrf(sid, action).encode())
        valid_choice = choice in ('yes', 'no') if action == 'logout' else choice == actions[action][1]
        if not valid_csrf or not valid_choice:
            if action == 'logout':
                # A stale tab, an earlier form version or a rotated session must
                # request fresh confirmation. This never authorizes the POST.
                return RedirectResponse(config.auth_url + '/account/logout', status_code=303)
            return HTMLResponse('Confirmation did not match.', status_code=403)
        try:
            session = await run_in_threadpool(manager.restore, sid)
        except AuthenticationError:
            if action == 'logout':
                # Logout is already complete when the server session is gone.
                response = RedirectResponse(config.app_url, status_code=303)
                clear(response, config.session_cookie)
                return response
            return HTMLResponse('Session expired. Sign in again.', status_code=401)
        if action == 'logout' and choice == 'no':
            return RedirectResponse(config.app_url, status_code=303)
        if action != 'logout':
            if time.time() - session.created_at > 600:
                return HTMLResponse('For deletion, sign in with Google again, then return here within 10 minutes.', status_code=403)
            try:
                def perform_cleanup():
                    if cleanup:
                        cleanup(session.user_id, action == 'delete')
                    else:
                        from .account_cleanup import admin_client, delete_account_data
                        delete_account_data(admin_client(config), session.user_id, delete_identity=action == 'delete')
                await run_in_threadpool(perform_cleanup)
                manager.vault.delete_user(session.user_id)
            except Exception:
                return HTMLResponse('Cleanup was not completed. The account remains locked if cleanup started. An operator must retry the pending cleanup job; no success has been assumed.', status_code=503)
        else:
            await run_in_threadpool(manager.logout, sid)
        response = RedirectResponse(config.app_url, status_code=303)
        clear(response, config.session_cookie)
        return response

    return app


def main():
    import uvicorn
    # Do not log callback query strings containing one-time authorization codes.
    uvicorn.run('hiking_time_predictor.auth_service:create_app', factory=True,
                host='127.0.0.1', port=8000, access_log=False)


if __name__ == '__main__':
    main()
