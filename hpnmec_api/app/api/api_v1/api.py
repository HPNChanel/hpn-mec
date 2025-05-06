from fastapi import APIRouter

# Import routers directly
from app.api.api_v1.endpoints.users import router as users_router
from app.api.api_v1.endpoints.doctors import router as doctors_router
from app.api.api_v1.endpoints.health_records import router as health_records_router
from app.api.api_v1.endpoints.appointments import router as appointments_router
from app.api.api_v1.endpoints.bookmarks import router as bookmarks_router
from app.api.api_v1.endpoints.auth import router as auth_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(doctors_router, prefix="/doctors", tags=["doctors"])
api_router.include_router(health_records_router, prefix="/health-records", tags=["health records"])
api_router.include_router(appointments_router, prefix="/appointments", tags=["appointments"])
api_router.include_router(bookmarks_router, prefix="/bookmarks", tags=["bookmarks"])