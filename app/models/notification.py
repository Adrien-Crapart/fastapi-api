from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, Boolean, String, DateTime, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base

if TYPE_CHECKING:
    from .user import User  # noqa: F401

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    message = Column(String, nullable=False)
    is_channel = Column(Boolean, default=False)
    users_list = Column(ARRAY(Integer))
    is_private = Column(Boolean, default=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="notifications")