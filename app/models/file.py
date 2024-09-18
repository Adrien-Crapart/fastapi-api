from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, text
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta

from app.db.base_class import Base

if TYPE_CHECKING:
    from .item import Item  # noqa: F401
    from .notification import Notification


class FileHistory(Base):
    __tablename__ = "file_history"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    operation = Column(String)  # "create", "change", "delete"
    user = Column(String)
    operation_date = Column(DateTime, default=datetime.utcnow)

class Trash(Base):
    __tablename__ = "trash"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    operation = Column(String)  # "delete" or "restore"
    user = Column(String)
    operation_date = Column(DateTime, default=datetime.utcnow)
    delete_after = Column(DateTime)