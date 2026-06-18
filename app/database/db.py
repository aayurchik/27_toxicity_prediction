from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings


class Base(DeclarativeBase):
    pass


# Синхронный движок (без aiosqlite)
engine = create_engine(
    settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://"),  # Убираем aiosqlite
    echo=False,
    future=True,
    connect_args={"check_same_thread": False}  # Для SQLite
)

# Синхронная фабрика сессий
SessionLocal = sessionmaker(
    engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


def init_db():
    """Создание таблиц при старте (синхронно)"""
    Base.metadata.create_all(bind=engine)


def close_db():
    """Закрытие соединения с БД"""
    engine.dispose()