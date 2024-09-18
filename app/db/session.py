from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SyncSession
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from app.core.config import settings

# Create a synchronous SQLAlchemy engine for your async session to use
sync_engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# Create an asynchronous SQLAlchemy engine and an asynchronous session class
async_engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    echo=settings.DEBUG,
    future=False,
)

# Create a synchronous session class for use in synchronous parts of your code
SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    class_=SyncSession,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)