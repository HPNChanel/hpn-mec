from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import all routers from the new routers/ directory
from app.routers import auth, users, doctors, appointments, bookmarks, health_records
from app.exceptions import UserAlreadyExistsException, InvalidCredentialsException, RecordNotFoundException

# (Future) Import custom exception handlers
# from app.exceptions import register_exception_handlers

app = FastAPI(
    title="HPN Medicare API",
    description="Healthcare Management System API",
    version="1.0.0",
)

# CORS middleware for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(doctors.router, prefix="/api/v1/doctors", tags=["doctors"])
app.include_router(appointments.router, prefix="/api/v1/appointments", tags=["appointments"])
app.include_router(bookmarks.router, prefix="/api/v1/bookmarks", tags=["bookmarks"])
app.include_router(health_records.router, prefix="/api/v1/health-records", tags=["health_records"])

# Register custom exception handlers
def exception_response(status: str, message: str):
    return JSONResponse(status_code=200, content={"status": status, "data": None, "message": message})

@app.exception_handler(UserAlreadyExistsException)
async def user_already_exists_handler(request: Request, exc: UserAlreadyExistsException):
    return JSONResponse(status_code=400, content={"status": "error", "data": None, "message": exc.detail})

@app.exception_handler(InvalidCredentialsException)
async def invalid_credentials_handler(request: Request, exc: InvalidCredentialsException):
    return JSONResponse(status_code=401, content={"status": "error", "data": None, "message": exc.detail})

@app.exception_handler(RecordNotFoundException)
async def record_not_found_handler(request: Request, exc: RecordNotFoundException):
    return JSONResponse(status_code=404, content={"status": "error", "data": None, "message": exc.detail})

# Register custom exception handlers (future)
# register_exception_handlers(app)

# Optional: async startup/shutdown events
@app.on_event("startup")
async def on_startup():
    # e.g., connect to external services, initialize resources
    pass

@app.on_event("shutdown")
async def on_shutdown():
    # e.g., cleanup resources
    pass

@app.get("/")
async def root():
    return {"message": "Welcome to HPN Medicare API. Access API at /api/v1"}