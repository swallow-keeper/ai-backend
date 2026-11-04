from __future__ import annotations

import base64
import hashlib
import json

from cryptography.fernet import Fernet


def fernet_from_secret(secret: str) -> Fernet:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def dump_secret(value: object, fernet: Fernet) -> str:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return fernet.encrypt(payload).decode("ascii")


def load_secret(token: str, fernet: Fernet) -> object:
    return json.loads(fernet.decrypt(token.encode("ascii")))
