from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ВОТ ЗДЕСЬ ДОЛЖНО БЫТЬ psycopg2, А НЕ pg8000!
SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://mts_user:mts_password@127.0.0.1:5433/mts_hackathon"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()