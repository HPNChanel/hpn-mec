import secrets
import base64

def generate_secret_key(length: int = 32) -> str:
    """Sinh ra một SECRET_KEY ngẫu nhiên, dùng cho thuật toán HS256"""
    key = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(key).decode('utf-8')

if __name__ == "__main__":
    secret_key = generate_secret_key(32)  # 256-bit key
    print(f"Generated SECRET_KEY: {secret_key}")
