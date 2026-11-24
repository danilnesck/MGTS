from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.api.endpoints import router as api_router

# Создаем таблицы при запуске (для хакатона ок, в продакшене - Alembic)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MTS Infrastructure Monitoring API",
    description="Backend для хакатона МТС (Computer Vision)",
    version="1.0.0"
)

# Разрешаем CORS, чтобы фронтенд мог стучаться
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "MTS Hackathon API is running. Go to /docs for Swagger."}