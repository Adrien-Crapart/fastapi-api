from typing import List

from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.crud.base import CRUDBase
from app.models.territory import City, Department
from app.schemas.territory import CityInDBBase, DepartmentInDBBase

class CRUDCity(CRUDBase[City, CityInDBBase, CityInDBBase]):
    def get_cities_by_prefix(
            self, db: Session, *, prefix: str
    ) -> List[City]:
      if len(prefix) >= 2:
        # Perform a case-insensitive search on city_name and city_postal fields
        return (
          db.query(self.model)
          .filter(
              or_(
                City.city_id.ilike(f"{prefix}%"), 
                City.city_name.ilike(f"%{prefix}%"), 
                City.city_postal.ilike(f"{prefix}%"),
                City.old_cities.ilike(f"%{prefix}%")
              )
            )
          .offset(0)
          .limit(10)
          .all()
        )
      
class CRUDDepartment(CRUDBase[Department, DepartmentInDBBase, DepartmentInDBBase]):      
    def get_department(
            self, db: Session, *, dep_id: str
    ) -> List[Department]:
      return (
        db.query(self.model)
        .filter(Department.dep_id == dep_id)
        .first()
      )
    
city = CRUDCity(City)
department = CRUDDepartment(Department)