"""Page registry for the cloud UI's top navigation.

st.Page objects carry per-run state (_can_be_called), so they are never shared
between sessions: main() builds fresh pages every run and keeps this session's
set in st.session_state, where in-page calls to action can find them.
Page callables read the per-run context from session state, so the Page
objects themselves never capture a user's repository or credentials.
"""
from dataclasses import dataclass

import streamlit as st

PAGES_KEY = '_nav_pages'
CONTEXT_KEY = '_page_ctx'
FLASH_KEY = '_flash'


@dataclass(frozen=True)
class PageSpec:
    key: str
    title: str
    icon: str
    url_path: str


SPECS = (
    PageSpec('home', 'Dashboard', ':material/space_dashboard:', 'dashboard'),
    PageSpec('plan', 'Plan a hike', ':material/route:', 'plan'),
    PageSpec('hikes', 'My hikes', ':material/hiking:', 'hikes'),
    PageSpec('pace', 'My pace', ':material/speed:', 'pace'),
    PageSpec('account', 'Account', ':material/account_circle:', 'account'),
)
SPEC = {spec.key: spec for spec in SPECS}


def build_pages(renderers):
    """Fresh st.Page objects for this run. ``renderers`` maps page key -> zero-arg callable."""
    return {spec.key: st.Page(renderers[spec.key], title=spec.title, icon=spec.icon,
                              url_path=spec.url_path, default=spec.key == 'home')
            for spec in SPECS}


def go(key):
    """Switch to a page from a button callback or page body (same session, no reload)."""
    if key == 'hikes':
        # Calls to action mean "the hike list", not whichever hike was open last.
        for stale in ('history_detail_hike_id', 'history_delete_hike'):
            st.session_state.pop(stale, None)
    st.switch_page(st.session_state[PAGES_KEY][key])


LAST_PAGE_KEY = '_last_page'


def arrived(current):
    """Track page changes; drop one-page view state (open hike, open dialog) when the user leaves My hikes."""
    previous = st.session_state.get(LAST_PAGE_KEY)
    st.session_state[LAST_PAGE_KEY] = current
    if previous is not None and previous != current:
        st.session_state.pop('history_delete_hike', None)
        if current != 'hikes':
            st.session_state.pop('history_detail_hike_id', None)


def nav_button(label, key, target, *, type='secondary', icon=None, width='content', help=None):
    """A button that opens another page. Uses a callback so it works before widgets render."""
    return st.button(label, key=key, type=type, icon=icon, width=width, help=help,
                     on_click=go, args=(target,))


def flash(message, icon=':material/check_circle:'):
    """Queue a toast for the next run (a toast emitted right before st.rerun() is lost)."""
    st.session_state[FLASH_KEY] = (message, icon)


def show_flash():
    pending = st.session_state.pop(FLASH_KEY, None)
    if pending:
        st.toast(pending[0], icon=pending[1], duration='long')


def context():
    return st.session_state[CONTEXT_KEY]
