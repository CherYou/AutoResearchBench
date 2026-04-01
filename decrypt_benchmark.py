import argparse
import base64
import gzip
import hashlib
import json
import os
from getpass import getpass
from pathlib import Path

try:
    from benchmark_crypto import (
        BenchmarkCryptoError,
        SCHEME_PASSWORD,
        SCHEME_PUBLIC,
        decrypt_payload,
        get_payload_scheme,
        load_payload,
        payload_requires_password,
        resolve_password,
    )
except ImportError:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    FORMAT_NAME = "scienceagent-benchmark-encrypted"
    FORMAT_VERSION = 1
    DEFAULT_PASSWORD_ENV = "SCIBENCH_BENCHMARK_PASSWORD"
    SCHEME_PASSWORD = "password_aes256gcm_scrypt"
    SCHEME_PUBLIC = "public_canary_xor_sha256"
    KEY_BYTES = 32

    class BenchmarkCryptoError(Exception):
        pass

    def _b64decode(value: str) -> bytes:
        return base64.b64decode(value.encode("ascii"))

    def _derive_key(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
        kdf = Scrypt(salt=salt, length=KEY_BYTES, n=n, r=r, p=p)
        return kdf.derive(password.encode("utf-8"))

    def _derive_public_mask(canary_string: str, length: int) -> bytes:
        digest = hashlib.sha256(canary_string.encode("utf-8")).digest()
        return digest * (length // len(digest)) + digest[: length % len(digest)]

    def _xor_bytes(data: bytes, mask: bytes) -> bytes:
        return bytes(a ^ b for a, b in zip(data, mask))

    def resolve_password(password=None, password_file=None, env_var=DEFAULT_PASSWORD_ENV, confirm=False):
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
        entered = getpass("Benchmark password: ")
        if not entered:
            raise BenchmarkCryptoError("Password cannot be empty.")
        if confirm:
            confirmed = getpass("Confirm benchmark password: ")
            if entered != confirmed:
                raise BenchmarkCryptoError("Passwords do not match.")
        return entered

    def _payload_without_ciphertext(payload):
        return {key: value for key, value in payload.items() if key != "ciphertext"}

    def _serialize_aad(payload):
        body = _payload_without_ciphertext(payload)
        return json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")

    def get_payload_scheme(payload):
        scheme = payload.get("scheme")
        if scheme:
            return scheme
        if payload.get("kdf") and payload.get("cipher"):
            return SCHEME_PASSWORD
        return SCHEME_PUBLIC

    def payload_requires_password(payload):
        return get_payload_scheme(payload) == SCHEME_PASSWORD

    def decrypt_payload(payload, password=None):
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

    def load_payload(input_path):
        return json.loads(Path(input_path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decrypt a public SciBench benchmark bundle back into the original file."
    )
    parser.add_argument("--input-file", required=True, help="Path to the encrypted bundle.")
    parser.add_argument(
        "--output-file",
        help="Path to the decrypted file. Defaults to the original filename stored in the bundle.",
    )
    parser.add_argument(
        "--password",
        help="Decryption password. Prefer --password-file or the environment variable for safety.",
    )
    parser.add_argument(
        "--password-file",
        help="Read the decryption password from a local file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.is_file():
        raise BenchmarkCryptoError(f"Input file not found: {input_path}")

    payload = load_payload(input_path)
    scheme = get_payload_scheme(payload)
    password = None
    if payload_requires_password(payload):
        password = resolve_password(
            password=args.password,
            password_file=args.password_file,
        )
    elif args.password or args.password_file:
        print("Warning: password arguments are ignored for public reversible obfuscation bundles.")
    plaintext = decrypt_payload(payload, password)

    metadata = payload.get("metadata", {})
    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
    else:
        original_name = metadata.get("original_filename")
        if not original_name:
            raise BenchmarkCryptoError("Encrypted bundle is missing the original filename metadata.")
        output_path = input_path.with_name(original_name)

    if output_path.exists() and not args.force:
        raise BenchmarkCryptoError(f"Output file already exists: {output_path}")

    output_path.write_bytes(plaintext)
    print(f"Decrypted file written to: {output_path}")
    print(f"Detected bundle scheme: {scheme}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkCryptoError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
