from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from typing import Generator
from app.core.config import settings

# Create SQLAlchemy engine with appropriate settings
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Verify connection before using from pool
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_size=10,        # Maximum number of connections in the pool
    max_overflow=20      # Maximum number of connections that can be created beyond pool_size
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy session.
    Yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()