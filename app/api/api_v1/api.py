from fastapi import APIRouter

from app.api.api_v1.endpoints import items, login, territories, users, utils, files, images, text, pdf, geometry

api_router = APIRouter()
# api_router.include_router(login.router, tags=["login"])
# api_router.include_router(files.router, prefix="/files", tags=["files"])
# api_router.include_router(images.router, prefix="/images", tags=["images"])
# api_router.include_router(text.router, prefix="/text", tags=["text"])
# api_router.include_router(pdf.router, prefix="/pdf", tags=["pdf"])
# api_router.include_router(geometry.router, prefix="/geometry", tags=["geometry"])
# api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
# api_router.include_router(items.router, prefix="/items", tags=["items"])
# api_router.include_router(territories.router, prefix="/cities", tags=["cities"])