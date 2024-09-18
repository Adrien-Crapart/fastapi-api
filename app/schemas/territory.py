from typing import Optional, List
from pydantic import BaseModel

class CityInDBBase(BaseModel):
    id: int
    city_id: str
    city_name: str
    city_postal: str
    old_cities: Optional[str] = None
    centroid: str
    bbox: str

    class Config:
        orm_mode = True

class DepartmentInDBBase(BaseModel):
    id: int
    dep_id: str
    name: str

    class Config:
        orm_mode = True