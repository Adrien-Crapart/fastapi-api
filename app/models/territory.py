# from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String
from geoalchemy2 import Geometry
# from sqlalchemy.orm import relationship

from app.db.base_class import Base

# if TYPE_CHECKING:
#     from .user import User  # noqa: F401


class City(Base):
    __tablename__ = "cities"
    __table_args__ = {"schema": "references"}

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(String)
    city_name = Column(String)
    city_postal = Column(String)
    old_cities = Column(String)
    centroid = Column(String)
    bbox = Column(String)
    # geom = Column(Geometry('MULTIPOLYGON'))

class Department(Base):
    __tablename__ = "departments"
    __table_args__ = {"schema": "references"}

    id = Column(Integer, primary_key=True, index=True)
    dep_id = Column(String)
    name = Column(String)