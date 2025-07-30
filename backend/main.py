from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from virtual_classroom.api import router as virtual_classroom_router
from virtual_classroom.dashboard_api import router as dashboard_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(virtual_classroom_router)
app.include_router(dashboard_router)