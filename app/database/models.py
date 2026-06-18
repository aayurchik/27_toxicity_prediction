from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.database.db import Base


class History(Base):
    __tablename__ = "History"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )
    time: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    endpoint: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )
    request_body: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    response_body: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    code: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )