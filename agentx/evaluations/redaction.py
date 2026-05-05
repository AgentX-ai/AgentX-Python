from __future__ import annotations

import re
from typing import Any, Dict

# Patterns that look like secrets
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),          # OpenAI / Anthropic style keys
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(r"(?i)(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*\S+"),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),      # long base64-like strings
]

_REDACTED = "[REDACTED]"


def redact_string(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(_REDACTED, value)
    return value


def redact_dict(obj: Any, _depth: int = 0) -> Any:
    if _depth > 10:
        return obj
    if isinstance(obj, dict):
        return {k: _redact_value(k, v, _depth) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_dict(item, _depth + 1) for item in obj]
    if isinstance(obj, str):
        return redact_string(obj)
    return obj


_SENSITIVE_KEYS = {
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "authorization", "auth", "cookie", "session", "credential",
    "private_key", "privatekey", "access_key", "accesskey",
}


def _redact_value(key: str, value: Any, depth: int) -> Any:
    if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
        return _REDACTED
    return redact_dict(value, depth + 1)
