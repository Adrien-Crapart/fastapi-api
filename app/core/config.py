import secrets
from decouple import config
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, HttpUrl, PostgresDsn, validator


class Settings(BaseSettings):
    API_V1_STR: str = config("API_V1_STR", default="/api/v1")
    SECRET_KEY: str = config("SECRET_KEY", default=secrets.token_urlsafe(32))
    DEBUG: int = 0
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=60, cast=int)
    SERVER_NAME: str = 'KHEOPS API'
    SERVER_HOST: AnyHttpUrl = 'http://localhost'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://127.0.0.1", 
        "http://127.0.0.1:4200", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    PROJECT_NAME: str = 'KHEOPS API'
    # SENTRY_DSN: Optional[HttpUrl] = None

    # @validator("SENTRY_DSN", pre=True)
    # def sentry_dsn_can_be_blank(cls, v: str) -> Optional[str]:
    #     if len(v) == 0:
    #         return None
    #     return v

    POSTGRES_HOST: str = config("POSTGRES_HOST")
    POSTGRES_PORT: str = config("POSTGRES_PORT")
    POSTGRES_USER: str = config("POSTGRES_USER")
    POSTGRES_PASSWORD: str = config("POSTGRES_PASSWORD")
    POSTGRES_DB: str = config("POSTGRES_DB")
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    SMTP_TLS: bool = config("SMTP_TLS", default=True, cast=bool)
    SMTP_PORT: Optional[int] = config("SMTP_PORT", default=None, cast=int)
    SMTP_HOST: Optional[str] = config("SMTP_HOST", default=None)
    SMTP_USER: Optional[str] = config("SMTP_USER", default=None)
    SMTP_PASSWORD: Optional[str] = config("SMTP_PASSWORD", default=None)
    EMAILS_FROM_EMAIL: Optional[EmailStr] = config("EMAILS_FROM_EMAIL", default=None)
    EMAILS_FROM_NAME: Optional[str] = config("EMAILS_FROM_NAME", default=None)

    @validator("EMAILS_FROM_NAME")
    def get_project_name(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if not v:
            return values["PROJECT_NAME"]
        return v

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48
    EMAIL_TEMPLATES_DIR: str = "/app/app/email-templates/build"
    EMAILS_ENABLED: bool = False

    @validator("EMAILS_ENABLED", pre=True)
    def get_emails_enabled(cls, v: bool, values: Dict[str, Any]) -> bool:
        return bool(
            values.get("SMTP_HOST")
            and values.get("SMTP_PORT")
            and values.get("EMAILS_FROM_EMAIL")
        )

    EMAIL_TEST_USER: EmailStr = config("EMAIL_TEST_USER")
    FIRST_SUPERUSER: EmailStr = config("FIRST_SUPERUSER")
    FIRST_SUPERUSER_PASSWORD: str = config("FIRST_SUPERUSER_PASSWORD")
    USERS_OPEN_REGISTRATION: bool = config("USERS_OPEN_REGISTRATION")

    class Config:
        case_sensitive = True


settings = Settings()
