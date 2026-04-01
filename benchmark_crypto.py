import base64
import gzip
import hashlib
import json
import os
import uuid
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


FORMAT_NAME = "scienceagent-benchmark-encrypted"
FORMAT_VERSION = 1
DEFAULT_PASSWORD_ENV = "SCIBENCH_BENCHMARK_PASSWORD"
SCHEME_PASSWORD = "password_aes256gcm_scrypt"
SCHEME_PUBLIC = "public_canary_xor_sha256"

KEY_BYTES = 32
NONCE_BYTES = 12
SALT_BYTES = 16

SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1


class BenchmarkCryptoError(Exception):
    pass


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _derive_key(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_BYTES, n=n, r=r, p=p)
    return kdf.derive(password.encode("utf-8"))


def resolve_password(
    password: Optional[str] = None,
    password_file: Optional[str] = None,
    env_var: str = DEFAULT_PASSWORD_ENV,
    confirm: bool = False,
) -> str:
    if password:
        return password

    if password_file:
        file_password = Path(password_file).read_text(encoding="utf-8").strip()
        if not file_password:
            raise BenchmarkCryptoError(f"Password file is empty: {password_file}")
        return file_password

    env_password = os.getenv(env_var)
    if env_password:
        return env_password

    prompt = "Benchmark password: "
    entered = getpass(prompt)
    if not entered:
        raise BenchmarkCryptoError("Password cannot be empty.")

    if confirm:
        confirmed = getpass("Confirm benchmark password: ")
        if entered != confirmed:
            raise BenchmarkCryptoError("Passwords do not match.")

    return entered


def _payload_without_ciphertext(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "ciphertext"}


def _serialize_aad(payload: Dict[str, Any]) -> bytes:
    body = _payload_without_ciphertext(payload)
    return json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def generate_canary_string(prefix: str = "scienceagent-benchmark") -> str:
    return f"{prefix}:{uuid.uuid4()}"


def _derive_public_mask(canary_string: str, length: int) -> bytes:
    digest = hashlib.sha256(canary_string.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def _xor_bytes(data: bytes, mask: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, mask))


def get_payload_scheme(payload: Dict[str, Any]) -> str:
    scheme = payload.get("scheme")
    if scheme:
        return scheme
    if payload.get("kdf") and payload.get("cipher"):
        return SCHEME_PASSWORD
    return SCHEME_PUBLIC


def payload_requires_password(payload: Dict[str, Any]) -> bool:
    return get_payload_scheme(payload) == SCHEME_PASSWORD


def build_encrypted_payload(
    plaintext: bytes,
    password: str,
    original_filename: str,
    canary_string: Optional[str] = None,
) -> Dict[str, Any]:
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    compressed = gzip.compress(plaintext, mtime=0)
    key = _derive_key(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)

    payload: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "scheme": SCHEME_PASSWORD,
        "kdf": {
            "name": "scrypt",
            "salt": _b64encode(salt),
            "n": SCRYPT_N,
            "r": SCRYPT_R,
            "p": SCRYPT_P,
        },
        "cipher": {
            "name": "AES-256-GCM",
            "nonce": _b64encode(nonce),
        },
        "compression": "gzip",
        "metadata": {
            "original_filename": original_filename,
            "original_size": len(plaintext),
            "original_sha256": hashlib.sha256(plaintext).hexdigest(),
            "canary_string": canary_string or generate_canary_string(),
        },
    }

    aad = _serialize_aad(payload)
    ciphertext = AESGCM(key).encrypt(nonce, compressed, aad)
    payload["ciphertext"] = _b64encode(ciphertext)
    return payload


def build_public_payload(
    plaintext: bytes,
    original_filename: str,
    canary_string: Optional[str] = None,
) -> Dict[str, Any]:
    public_canary = canary_string or generate_canary_string()
    compressed = gzip.compress(plaintext, mtime=0)
    mask = _derive_public_mask(public_canary, len(compressed))
    ciphertext = _xor_bytes(compressed, mask)

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "scheme": SCHEME_PUBLIC,
        "public_obfuscation": {
            "name": "xor-sha256",
        },
        "compression": "gzip",
        "metadata": {
            "original_filename": original_filename,
            "original_size": len(plaintext),
            "original_sha256": hashlib.sha256(plaintext).hexdigest(),
            "canary_string": public_canary,
        },
        "ciphertext": _b64encode(ciphertext),
    }


def decrypt_payload(payload: Dict[str, Any], password: Optional[str] = None) -> bytes:
    if payload.get("format") != FORMAT_NAME:
        raise BenchmarkCryptoError("Unsupported encrypted file format.")
    if payload.get("version") != FORMAT_VERSION:
        raise BenchmarkCryptoError(f"Unsupported encrypted file version: {payload.get('version')}")
    if payload.get("compression") != "gzip":
        raise BenchmarkCryptoError(f"Unsupported compression: {payload.get('compression')}")

    scheme = get_payload_scheme(payload)

    if scheme == SCHEME_PUBLIC:
        metadata = payload.get("metadata", {})
        canary_string = metadata.get("canary_string")
        if not canary_string:
            raise BenchmarkCryptoError("Public obfuscation payload is missing the canary string.")

        ciphertext = _b64decode(payload["ciphertext"])
        mask = _derive_public_mask(canary_string, len(ciphertext))
        compressed = _xor_bytes(ciphertext, mask)
        plaintext = gzip.decompress(compressed)
    elif scheme == SCHEME_PASSWORD:
        if not password:
            raise BenchmarkCryptoError("This payload requires a password.")

        kdf_config = payload.get("kdf", {})
        cipher_config = payload.get("cipher", {})

        if kdf_config.get("name") != "scrypt":
            raise BenchmarkCryptoError("Unsupported KDF configuration.")
        if cipher_config.get("name") != "AES-256-GCM":
            raise BenchmarkCryptoError("Unsupported cipher configuration.")

        salt = _b64decode(kdf_config["salt"])
        nonce = _b64decode(cipher_config["nonce"])
        ciphertext = _b64decode(payload["ciphertext"])
        key = _derive_key(
            password=password,
            salt=salt,
            n=int(kdf_config["n"]),
            r=int(kdf_config["r"]),
            p=int(kdf_config["p"]),
        )

        aad = _serialize_aad(payload)

        try:
            compressed = AESGCM(key).decrypt(nonce, ciphertext, aad)
        except Exception as exc:
            raise BenchmarkCryptoError("Failed to decrypt payload. Check the password and file integrity.") from exc

        plaintext = gzip.decompress(compressed)
    else:
        raise BenchmarkCryptoError(f"Unsupported payload scheme: {scheme}")

    metadata = payload.get("metadata", {})
    expected_sha256 = metadata.get("original_sha256")
    actual_sha256 = hashlib.sha256(plaintext).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise BenchmarkCryptoError("Decrypted content failed integrity verification.")

    return plaintext


def load_payload(input_path: Path) -> Dict[str, Any]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def save_payload(payload: Dict[str, Any], output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise BenchmarkCryptoError(f"Output file already exists: {output_path}")
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
