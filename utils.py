import re

def validate_structure(log) -> int:
    """Check the generated logs with regex for structural and format deviations."""
    log_penalty = 0

    # Remove last pipe
    log = log[:-1]

    if any(ord(c) > 127 for c in log):
        # print("Non-ASCII characters detected")
        log_penalty += 1

    validation_patterns = {
        # host: domain names, IPv4 dotted‐quad, optional port
        'host': r'^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))+(?::\d{1,5})?$',
        # Autonomous system number that ends in .0
        'as_number': r'^\d+\.0$',
        # Well‑formed absolute path with one or more segments, optionally ending in a slash.
        'path': r'^(?:None|/(?:[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)*)?/?)$',
        # Matches either None, or any HTTP(S) URL, or bare domain/IP (with optional port and path):
        'referer': r'^(?:None|(?:https?://)?[A-Za-z0-9.-]+(?::\d{1,5})?(?:/[^\s]*)?)$',
        # Response/request sizes: non‐negative integers
        'response_size': r'^\d+$',
        'request_size': r'^\d+$',
        # Client address: internal numerical representation, non-negative integer
        'client_address': r'^\d+$',
        # None/nan, MIME type/subtype consisting only of ASCII, digits, and the common extra symbols
        'content_type': r'^(?:None|nan|[A-Za-z0-9!#$&^_.+\-]+/[A-Za-z0-9!#$&^_.+\-]+(?:;\s*charset=[A-Za-z0-9._\-]+)?)$',
        # Matches any ASCII user‑agent string of max 512 characters
        'user_agent': r'^(?=.{2,512}$)[\x20-\x7E]+$',
        # Matches exactly three digits, 100–599
        'status_code': r'^[1-5]\d{2}$',
        # Free‐form string of printable ASCII (max length 100)
        'as_organization': r'^(?:None|[\x20-\x7E]{1,100})$'
        }

    allowed_values = {
        'method': {'GET', 'HEAD', 'POST', 'OPTIONS', 'PUT', 'DELETE', 
                   'PATCH', 'PROPFIND', 'TRACK', 'FLURP', 'None'}}

    required_keys = [
        'host', 'as_number', 'client_address', 'method', 'path',
        'request_size', 'referer', 'content_type', 'status_code',
        'user_agent', 'as_organization']

    # Check field count
    parts = log.split('|')
    if len(parts) != len(required_keys):
        log_penalty += 1
        # print('Invalid field count:', len(parts))

    # Parse key:value pairs
    key_values = {}
    for part in parts:
        if ':' not in part:
            # print('Invalid key‑value format:', part)
            log_penalty += 1
            continue
        key, value = part.split(':', 1)
        key = key.strip()
        value = value.strip()

        if key not in required_keys:
            # print(f'Unexpected field "{key}"')
            log_penalty += 1
        else:
            key_values[key] = value

    # Field‑specific checks
    for field in required_keys:
        val = key_values.get(field)
        if val is None:
            # print(f'Missing field: {field}')
            log_penalty += 1
            continue

        if field == 'method':
            if val not in allowed_values['method']:
                # print('Invalid HTTP method:', val)
                log_penalty += 1
        else:
            pattern = validation_patterns[field]
            if not re.match(pattern, val):
                # print(f'Field "{field}" failed regex: {val!r}')
                log_penalty += 1

    return log_penalty