from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import BASE_DIR, CONFIG

DATABASE_URL = CONFIG['database_url']
instance_dir = BASE_DIR / 'instance'
instance_dir.mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
