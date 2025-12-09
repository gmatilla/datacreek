import os

try:
    from flask_login import UserMixin
except Exception:  # pragma: no cover - optional dependency

    class UserMixin:  # type: ignore[misc]
        """Fallback when Flask-Login isn't installed."""

        pass


from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from datacreek.utils.config import load_config


def get_database_url() -> str:
    """Return DB connection string from env or config."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    try:
        cfg = load_config()
        return cfg.get("database", {}).get("url", "sqlite:////tmp/datacreek.db")
    except Exception:
        return "sqlite:////tmp/datacreek.db"


DATABASE_URL = get_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base, UserMixin):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False, default="")

    sources = relationship("SourceData", back_populates="owner")
    datasets = relationship("Dataset", back_populates="owner")


class SourceData(Base):
    __tablename__ = "sources"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    path = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    # optional serialized lists of entities and facts extracted during ingest
    entities = Column(Text, nullable=True)
    facts = Column(Text, nullable=True)

    owner = relationship("User", back_populates="sources")
    datasets = relationship("Dataset", back_populates="source")


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    path = Column(String, nullable=False)
    content = Column(Text, nullable=True)

    owner = relationship("User", back_populates="datasets")
    source = relationship("SourceData", back_populates="datasets")


class TenantPrivacy(Base):
    """Differential privacy budget per tenant."""

    __tablename__ = "tenant_privacy"
    tenant_id = Column(Integer, primary_key=True, index=True)
    epsilon_used = Column(Float, nullable=False, default=0.0)
    epsilon_max = Column(Float, nullable=False, default=0.0)


def init_db() -> None:
    """Create database tables."""
    Base.metadata.create_all(bind=engine)


def verify_password(user: User, password: str) -> bool:
    """Return True if the password matches."""
    from werkzeug.security import check_password_hash

    return check_password_hash(user.password_hash, password)
