"""Approximate sunrise and sunset for daylight planning; no external service.

Implements the standard sunrise equation (as on NOAA's solar calculator and
Wikipedia's "Sunrise equation"). It is accurate to a minute or two at hiking
latitudes, which is plenty for "will I finish before dark" planning.
"""
from datetime import date, datetime, timedelta, timezone
from math import acos, asin, cos, degrees, isfinite, radians, sin

_J2000 = 2451545.0
_EPOCH = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


def _julian(day: date) -> float:
    return day.toordinal() + 1721424.5


def _to_datetime(julian: float) -> datetime:
    return _EPOCH + timedelta(days=julian - _J2000)


def sun_times(latitude: float, longitude: float, day: date) -> tuple[datetime | None, datetime | None]:
    """UTC sunrise and sunset on the local calendar ``day`` at a location.

    Longitude is positive east. Returns (None, None) during polar day or night,
    or for invalid coordinates.
    """
    if not (isfinite(latitude) and isfinite(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None, None
    n = round(_julian(day) - _J2000 + 0.0008)
    mean_noon = n - longitude / 360
    anomaly = radians((357.5291 + 0.98560028 * mean_noon) % 360)
    center = 1.9148 * sin(anomaly) + 0.0200 * sin(2 * anomaly) + 0.0003 * sin(3 * anomaly)
    ecliptic = radians((degrees(anomaly) + center + 180 + 102.9372) % 360)
    transit = _J2000 + mean_noon + 0.0053 * sin(anomaly) - 0.0069 * sin(2 * ecliptic)
    declination = asin(sin(ecliptic) * sin(radians(23.4397)))
    phi = radians(latitude)
    cos_hour = (sin(radians(-0.833)) - sin(phi) * sin(declination)) / (cos(phi) * cos(declination))
    if not -1 <= cos_hour <= 1:
        return None, None
    half_day = degrees(acos(cos_hour)) / 360
    return _to_datetime(transit - half_day), _to_datetime(transit + half_day)


def sunset(latitude: float, longitude: float, day: date) -> datetime | None:
    return sun_times(latitude, longitude, day)[1]
