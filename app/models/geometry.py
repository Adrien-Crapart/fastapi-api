from pydantic import BaseModel

class Location(BaseModel):
    latitude: float
    longitude: float

class POI(BaseModel):
    id: int
    name: str
    location: Location