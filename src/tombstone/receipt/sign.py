"""Ed25519 signing of receipts. Keys live in ``.tombstone/keys/`` (0600, gitignored)."""

from __future__ import annotations

from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from tombstone.errors import SignatureInvalid
from tombstone.util import secure_write


def generate_keypair(private_path: Path, public_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    private_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    secure_write(private_path, priv, 0o600)
    secure_write(public_path, pub, 0o644)


def load_private_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SignatureInvalid(f"{path} is not an Ed25519 private key")
    return key


def load_public_key(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise SignatureInvalid(f"{path} is not an Ed25519 public key")
    return key


def public_key_hex(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def public_key_from_hex(hex_key: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(hex_key))


def sign_bytes(key: Ed25519PrivateKey, payload: bytes) -> str:
    return key.sign(payload).hex()


def verify_bytes(public: Ed25519PublicKey, payload: bytes, signature_hex: str) -> None:
    """Raises ``SignatureInvalid`` if the signature does not verify."""
    try:
        public.verify(bytes.fromhex(signature_hex), payload)
    except (InvalidSignature, ValueError) as e:
        raise SignatureInvalid("receipt signature does not verify") from e
