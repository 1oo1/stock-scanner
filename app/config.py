# Configuration settings for the application

from datetime import timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_llm_config():
    """
    Load LLM configuration from environment variables.
    Returns:
        dict: LLM configuration settings: API_KEY, API_URL, API_MODEL, API_TIMEOUT
    """

    return {
        "API_KEY": os.getenv("API_KEY"),
        "API_URL": os.getenv("API_URL"),
        "API_MODEL": os.getenv("API_MODEL"),
        "API_TIMEOUT": int(os.getenv("API_TIMEOUT", "60")),
    }


def get_flask_config():
    """
    Load Flask configuration from environment variables.
    Returns:
        dict: Flask configuration settings.
    """

    return {
        # flask
        "DEBUG": os.getenv("FLASK_DEBUG", "False") == "True",
        "PORT": int(os.getenv("FLASK_PORT", "8888")),
        "HOST": os.getenv("FLASK_HOST", "127.0.0.1"),
        # jwt
        "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY"),
        "JWT_ACCESS_TOKEN_EXPIRES": timedelta(
            hours=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_HOURS", 1))
        ),
        "JWT_TOKEN_LOCATION": ["cookies"],
        "JWT_COOKIE_SECURE": False,
        "JWT_COOKIE_CSRF_PROTECT": True,
        "JWT_COOKIE_SAMESITE": "Lax",
        # postgresql
        "SQLALCHEMY_DATABASE_URI": f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}",
        "SQLALCHEMY_BINDS": {},
    }
