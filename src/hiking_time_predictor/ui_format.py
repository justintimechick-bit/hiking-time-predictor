"""Hiker-facing labels and number formats for the cloud UI.

Display only: stored codes ('day_hike', 'stationary_break', ...) never change.
Every function here is pure so it can be unit-tested without Streamlit.
"""
from datetime import date, datetime, time, timedelta, timezone
from math import isfinite
import re

import pandas as pd

from .modeling import MODEL_LABELS, NAISMITH_LABEL

RULE_NAME = 'standard hiking rule'
RULE_DEFINITION = '3 mph plus 1 hour for every 2,000 ft of climbing'

HIKE_TYPE_LABELS = {'day_hike': 'Day hike', 'backpacking': 'Backpacking'}
DIRECTION_LABELS = {'one-way': 'One way', 'out-and-back': 'Out and back'}

# Break-event classifications, as hikers would describe them.
CLASSIFICATION_LABELS = {
    'moving': 'Moving',
    'stationary_break': 'Rest break',
    'short_stop': 'Short pause',
    'unknown': 'Unclear GPS time',
    'camping': 'Camp / overnight',
    'non_hiking': 'Not part of the hike',
}

# Recording-level GPS quality flags (gpx.py) and per-stop flags (breaks.py).
FLAG_LABELS = {
    'missing_hike_type': 'Hike type not set',
    'recording_gap': 'GPS gaps',
    'possible_rest_during_gap': 'Possible rest during a GPS gap',
    'implausible_speed': 'GPS jumps',
    'ambiguous_slow_movement': 'Very slow sections',
    'missing_elevation': 'Missing elevation',
    'duplicate_timestamp': 'Repeated timestamps',
    'missing_timestamp': 'Incomplete timing',
    'insufficient_timestamps': 'Incomplete timing',
    'incomplete_elapsed_time': 'Incomplete timing',
    'naive_timestamp': 'Timestamps without a time zone',
    'invalid_coordinate': 'Invalid GPS points',
    'track_boundary': 'Several track segments',
    'unknown_interval': 'Unclear GPS time',
    'no_measured_movement': 'No measured movement',
    'gps_heuristic': 'Detected from GPS',
    'hiker_confirmed': 'Confirmed by you',
    'manual_review': 'Reviewed by you',
    'stationary_cluster_drift': 'GPS drifted during the stop',
    'exceeds_ordinary_break_duration': 'Longer than a normal break',
    'extended_stationary_camping_candidate': 'Very long stop (possible camp)',
    'unresolved_recording_interval': 'Unclear GPS interval',
}
# Flags that describe how a stop was found, not a problem with the recording.
INFORMATIONAL_FLAGS = {'gps_heuristic', 'hiker_confirmed', 'manual_review'}

MOVING_MODEL_NAMES = {
    'A': 'distance',
    'B': 'distance and climbing',
    'C': 'distance, climbing and hike type',
}
# Rows of the saved evaluation tables, renamed for display.
METRIC_MODEL_NAMES = {
    MODEL_LABELS['A']: 'Your pace model: distance only',
    MODEL_LABELS['B']: 'Your pace model: distance + climbing',
    MODEL_LABELS['C']: 'Your pace model: distance + climbing + hike type',
    NAISMITH_LABEL: 'Standard hiking rule',
}
BREAK_METHOD_NAMES = {
    'personal_minutes_per_mile': 'Your stop-per-mile rate',
    'A': 'Distance regression',
    'B': 'Distance + climbing regression',
    'C': 'Distance + climbing + hike type regression',
}

_MARKDOWN_SPECIAL = re.compile(r'([\\`*_{}\[\]()#+\-.!|<>~$:])')


def md_escape(text) -> str:
    """Render user text (hike and route names, display names) literally in Markdown.

    Streamlit Markdown also understands :color[...] and :material/...: directives,
    so colons are escaped along with the usual Markdown punctuation.
    """
    return _MARKDOWN_SPECIAL.sub(r'\\\1', '' if text is None else str(text))


def flag_labels(flags, *, include_informational=False) -> list[str]:
    """Unique, ordered friendly labels for a ';'-separated flag string."""
    labels = []
    for flag in str(flags or '').split(';'):
        flag = flag.strip()
        if not flag or (flag in INFORMATIONAL_FLAGS and not include_informational):
            continue
        label = FLAG_LABELS.get(flag, flag.replace('_', ' ').capitalize())
        if label not in labels:
            labels.append(label)
    return labels


def _finite(value):
    try:
        return value is not None and not pd.isna(value) and isfinite(float(value))
    except (TypeError, ValueError):
        return False


def miles(value, decimals=1) -> str:
    return f'{float(value):,.{decimals}f} mi' if _finite(value) else '—'


def feet(value) -> str:
    return f'{float(value):,.0f} ft' if _finite(value) else '—'


def pace(minutes_per_mile) -> str:
    """Pace as 'mm:ss /mi' (24.41 -> '24:25 /mi')."""
    if not _finite(minutes_per_mile) or float(minutes_per_mile) <= 0:
        return '—'
    seconds = int(round(float(minutes_per_mile) * 60))
    return f'{seconds // 60}:{seconds % 60:02d} /mi'


def speed(mph) -> str:
    return f'{float(mph):.1f} mph' if _finite(mph) else '—'


def clock(moment) -> str:
    """'8:05 AM' without platform-specific strftime flags."""
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def parse_time(value):
    """A timezone-aware timestamp from an ISO string, or None."""
    if not value:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp


def friendly_date(value, tz=None, *, weekday=False) -> str:
    """'Oct 21, 2026' (or 'Sat, Oct 21, 2026') in the viewer's timezone when known."""
    stamp = parse_time(value)
    if stamp is None:
        return str(value or '—')[:10]
    if tz:
        try:
            stamp = stamp.tz_convert(tz)
        except (TypeError, ValueError, KeyError):
            pass
    text = f'{stamp.strftime("%b")} {stamp.day}, {stamp.year}'
    return f'{stamp.strftime("%a")}, {text}' if weekday else text


def relative_day(value, now=None, tz=None) -> str:
    """'today', 'yesterday', '3 days ago', or a date for anything older than a week."""
    stamp = parse_time(value)
    if stamp is None:
        return str(value or '—')[:10]
    now = parse_time(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if tz:
        try:
            stamp, now = stamp.tz_convert(tz), now.tz_convert(tz)
        except (TypeError, ValueError, KeyError):
            pass
    days = (now.normalize() - stamp.normalize()).days
    if days <= 0:
        return 'today'
    if days == 1:
        return 'yesterday'
    if days < 7:
        return f'{days} days ago'
    return friendly_date(stamp)


def method_name(forecast_label) -> str:
    """The primary-estimate label from predictor.forecast_route, for hikers."""
    for key, label in MODEL_LABELS.items():
        if forecast_label == label:
            return f'Your pace model ({MOVING_MODEL_NAMES[key]})'
    return 'Standard hiking rule'


def finish_time(start: datetime, total_minutes) -> tuple[datetime, int] | None:
    """Expected finish and how many days later it falls, for a planned start."""
    if start is None or not _finite(total_minutes):
        return None
    finish = start + timedelta(minutes=float(total_minutes))
    return finish, (finish.date() - start.date()).days


def combine_start(day: date | None, start: time | None, tz=None) -> datetime | None:
    if start is None:
        return None
    day = day or date.today()
    moment = datetime.combine(day, start)
    if tz:
        try:
            from zoneinfo import ZoneInfo
            return moment.replace(tzinfo=ZoneInfo(tz))
        except Exception:
            pass
    return moment.replace(tzinfo=timezone.utc) if tz else moment


_RANGE_NAMES = {'distance_miles': 'distance', 'elevation_gain_ft': 'climbing',
                'elevation_loss_ft': 'descent'}
_RANGE_WARNING = re.compile(
    r'^(?:(?P<leg>Outbound|Return): )?(?P<scope>Route total|Segment) (?P<field>\w+) outside training range '
    r'\[(?P<low>[-\d.]+), (?P<high>[-\d.]+)\]')


def humanize_note(note: str) -> str:
    """Readable versions of predictor/route notes; unknown notes pass through unchanged."""
    text = str(note)
    match = _RANGE_WARNING.match(text)
    if match:
        name = _RANGE_NAMES.get(match['field'], match['field'].replace('_', ' '))
        unit = miles if match['field'] == 'distance_miles' else feet
        span = f'{unit(float(match["low"]))} to {unit(float(match["high"]))}'
        leg = f"{match['leg']} leg: " if match['leg'] else ''
        if match['scope'] == 'Segment':
            return (f'{leg}some half-mile sections have more {name} than any section of the hikes '
                    f'you\'ve trained on ({span}), so those sections are rough estimates.')
        return (f'{leg}this route\'s total {name} is outside the range of hikes you\'ve trained on '
                f'({span}), so treat the estimate as rough.')
    if text.startswith('GPX timestamps are ignored'):
        return 'Only distance and elevation from your file are used. Any times recorded in it are ignored.'
    if text.startswith('No trained model loaded'):
        return f'Estimated with the {RULE_NAME} ({RULE_DEFINITION}).'
    if text.startswith('Training range unavailable'):
        return 'Train your pace model to check this route against the hikes you\'ve recorded.'
    if text.startswith('Whole-route training range unavailable'):
        return ('None of your recorded hikes was complete enough to compare whole-route size, '
                'so only half-mile sections were checked.')
    hike_type = re.match(r"Hike type '(\w+)' was not observed", text)
    if hike_type:
        label = HIKE_TYPE_LABELS.get(hike_type[1], hike_type[1]).lower()
        return f'You haven\'t trained on any {label} trips yet.'
    for field, name in _RANGE_NAMES.items():
        text = text.replace(field, name)
    text = re.sub(r'\s*\(\d+ input\(s\)\)', '', text)
    return re.sub(r'(\d+\.\d{2})\d+', r'\1', text)


def route_extremes(route_totals: dict, route_ranges: dict) -> list[str]:
    """Plain warnings when a planned route is bigger or smaller than any trained hike."""
    messages = []
    for field, name in (('elevation_gain_ft', 'climb'), ('distance_miles', 'distance')):
        bounds, value = route_ranges.get(field), route_totals.get(field)
        if not bounds or not _finite(value):
            continue
        low, high = float(bounds[0]), float(bounds[1])
        fmt = miles if field == 'distance_miles' else feet
        if float(value) > high * 1.05:
            messages.append(f'Bigger {name} than any hike you\'ve trained on '
                            f'({fmt(value)} vs. your longest {fmt(high)})')
        elif float(value) < low * 0.95:
            messages.append(f'Smaller {name} than any hike you\'ve trained on '
                            f'({fmt(value)} vs. your shortest {fmt(low)})')
    return messages


def humanize_filename(filename: str) -> str:
    """'Mt_Diablo_summit-loop.gpx' -> 'Mt Diablo summit loop'."""
    stem = re.sub(r'\.gpx$', '', str(filename or ''), flags=re.IGNORECASE)
    return re.sub(r'[_\s]+|(?<=\w)-(?=\w)', ' ', stem).strip() or 'Recorded hike'


def safe_filename(name: str, suffix: str) -> str:
    stem = re.sub(r'[^\w.-]+', '_', str(name or '')).strip('._')[:80]
    return f'{stem or "hike"}{suffix}'
