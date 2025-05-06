import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    PROJECT_NAME: str = "HPN Medicare API"
    PROJECT_VERSION: str = "1.0.0"
    
    # Database settings
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "HPNChanel1312$")
    MYSQL_SERVER: str = os.getenv("MYSQL_SERVER", "localhost")
    MYSQL_PORT: str = os.getenv("MYSQL_PORT", "3306")
    MYSQL_DB: str = os.getenv("MYSQL_DB", "hpnmec")
    
    DATABASE_URL: str = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_SERVER}:{MYSQL_PORT}/{MYSQL_DB}"

    # JWT settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "3A8Io_w33JMvxf7_Lyla5MFBQKW_2HqgbbvG_u0O-Xs=")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # 1 day

settings = Settings()