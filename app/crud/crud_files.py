from typing import Any, Dict, Optional, Union
import shutil

from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password
# from app.crud.base import CRUDBase
# from app.models.user import User
# from app.schemas.user import UserCreate, UserUpdate


class CRUDFile():
    def get_file(file: bytes):
        content = file.decode('utf-8')
        lines = content.split('\n')
        return {'lines': lines}
    
    def get_uploadfile(upload_file: bytes):
        path = f"files/{upload_file.filename}"
        with open(path, 'w+b') as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return {
            'filename': path,
            'type': upload_file.content_type
        }
    
    def get_downloadfile(name: str):
        path = f"files/{name}"
        return path
    
file = CRUDFile()