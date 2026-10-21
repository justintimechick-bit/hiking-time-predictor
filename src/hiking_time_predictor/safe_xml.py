"""Bounded XML validation before passing untrusted bytes to existing GPX readers."""
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

MAX_GPX_BYTES = 20 * 1024 * 1024
MAX_GPX_POINTS = 100_000


def validate_gpx_xml(contents):
    if not isinstance(contents, bytes) or not 0 < len(contents) <= MAX_GPX_BYTES:
        raise ValueError('GPX uploads must contain 1 byte to 20 MiB.')
    try:
        text = contents.decode('utf-8-sig')
        root = ElementTree.fromstring(text, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (UnicodeError, ElementTree.ParseError, DefusedXmlException):
        raise ValueError('Cannot parse GPX: use valid UTF-8 XML without DTDs or entities.') from None
    if root.tag.split('}')[-1].lower() != 'gpx':
        raise ValueError('The XML root must be GPX.')
    if sum(node.tag.split('}')[-1] in ('trkpt', 'rtept', 'wpt') for node in root.iter()) > MAX_GPX_POINTS:
        raise ValueError('GPX uploads are limited to 100,000 points. Split this recording first.')
