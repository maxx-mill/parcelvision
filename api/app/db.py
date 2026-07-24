from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings

_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    """Bring the schema to head via alembic at API startup. Databases created
    by the pre-alembic `create_all` era are adopted by stamping head."""
    from alembic import command
    from alembic.config import Config

    api_root = Path(__file__).resolve().parents[1]
    cfg = Config(str(api_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(api_root / "alembic"))

    insp = inspect(get_engine())
    if insp.has_table("jobs") and not insp.has_table("alembic_version"):
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")


def get_session() -> Generator[Session, None, None]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
