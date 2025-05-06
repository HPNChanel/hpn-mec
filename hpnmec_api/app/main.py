from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.api_v1.api import api_router
from app.models.base import Base
from app.db.database import engine

app = FastAPI(
    title="HPN Medicare API",
    description="Healthcare Management System API",
    version="1.0.0",
)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Configure CORS as per requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to HPN Medicare API. Access API at /api/v1"}