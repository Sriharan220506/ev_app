"""
Database connection and session management
Uses PostgreSQL on Railway
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# Get database URL from Railway environment variable
# Railway automatically sets DATABASE_URL when you add PostgreSQL
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

# Fix for Railway PostgreSQL URL (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine
engine = create_engine(DATABASE_URL, echo=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session - use as dependency in FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
