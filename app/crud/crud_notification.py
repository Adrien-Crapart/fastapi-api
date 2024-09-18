from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.notification import Notification
from app.schemas.notification import NotificationInDBBase


class CRUDNotification(CRUDBase[Notification, NotificationInDBBase, NotificationInDBBase]):
    def create_notification(
        self, db: Session, *, user_id: int, message: str
    ) -> Notification:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        notification = Notification(user_id=user_id, message=message)
        db.add(notification)
        db.commit()
        db.refresh(notification)
        return notification
    
    def get_notification(
        notification_id: int,
        db: Session = Depends(get_db)
    ):
        notification = db.query(Notification).filter(Notification.id == notification_id).first()
        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return notification
    
    def get_all_notifications(
        user_id: int,
        db: Session = Depends(get_db)
    ):
        notifications = db.query(Notification).filter(Notification.user_id == user_id).all()
        return notifications
    
    def mark_notification_as_read(
        notification_id: int,
        db: Session = Depends(get_db)
    ):
        notification = db.query(Notification).filter(Notification.id == notification_id).first()
        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        notification.is_read = True
        db.commit()
        db.refresh(notification)
        return notification

item = CRUDNotification(Notification)