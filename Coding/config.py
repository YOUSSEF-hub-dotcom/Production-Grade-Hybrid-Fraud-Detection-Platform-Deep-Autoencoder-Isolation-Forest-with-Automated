import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    DATABASE_URL: str = os.getenv("DATABASE_URL")

    MODEL_URI: str = os.getenv("MODEL_URI")
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

settings = Settings()