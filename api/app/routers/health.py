from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import get_session
from ..queue import get_redis
from ..schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(session: Session = Depends(get_session)) -> HealthOut:
    db_ok = redis_ok = False
    try:
        session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    try:
        redis_ok = bool(get_redis().ping())
    except Exception:
        pass
    return HealthOut(status="ok" if db_ok and redis_ok else "degraded", db=db_ok, redis=redis_ok)
