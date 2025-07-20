from fastapi import FastAPI
from virtual_classroom.api import router as virtual_classroom_router

app = FastAPI()
app.include_router(virtual_classroom_router)