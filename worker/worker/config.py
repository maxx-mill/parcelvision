import os


def database_url() -> str:
    return (
        f"postgresql+psycopg://{os.environ.get('POSTGRES_USER', 'parcelvision')}"
        f":{os.environ.get('POSTGRES_PASSWORD', 'parcelvision-dev-only')}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ.get('POSTGRES_DB', 'parcelvision')}"
    )


def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def inference_backend() -> str:
    return os.environ.get("INFERENCE_BACKEND", "local_cpu")


def naip_year() -> int | None:
    raw = os.environ.get("NAIP_YEAR", "").strip()
    return int(raw) if raw else None


def imagery_dir() -> str:
    return os.environ.get("IMAGERY_DIR", "/data/imagery")
