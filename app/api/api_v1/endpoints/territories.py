from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.get("/cities", response_model=List[schemas.CityInDBBase])
async def read_cities(
    db: Session = Depends(deps.get_sync_session),
    prefix: str = None,
    # current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve cities. The limit as 10 results maximum.
    """
    cities = crud.city.get_cities_by_prefix(
        db=db, prefix=prefix
    )
    return cities

@router.get("/departments", response_model=List[schemas.DepartmentInDBBase])
async def read_departments(
    db: Session = Depends(deps.get_sync_session),
    skip: int = 0,
    limit: int = 100,
    # current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve all departments in France and DROM.
    """
    departments = crud.department.get_multi(db, skip=skip, limit=limit)
    return departments