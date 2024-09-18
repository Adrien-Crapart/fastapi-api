from pydantic import BaseModel


class Msg(BaseModel):
    msg: str

class NotificationInDBBase(BaseModel):
    id: int
    user_id: int
    message: int
    is_channel = bool
    users_list = list
    is_private = bool
    is_read = bool
    created_at = str

    class Config:
        orm_mode = True