import os
import logging
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from datacreek.utils.config import load_config

LOGGER = logging.getLogger(__name__)


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


def _validate_database_url(url: str) -> str:
    if not url:
        raise ValueError("DATABASE_URL must not be empty")
    if "://" not in url:
        raise ValueError("DATABASE_URL must look like a SQLAlchemy URL")
    return url


DATABASE_URL: str | None = None
Base = declarative_base()

_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def get_engine() -> Engine:
    """Create and cache the SQLAlchemy engine lazily."""

    global _engine, DATABASE_URL
    if _engine is None:
        url = get_database_url()
        url = _validate_database_url(url)
        DATABASE_URL = url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = sqlalchemy_create_engine(url, connect_args=connect_args)
        LOGGER.debug("Created SQLAlchemy engine for %s", url)
    return _engine


def get_session_factory() -> sessionmaker:
    """Return a sessionmaker bound to the lazily created engine."""

    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _session_factory


class _LazySessionFactory:
    """Thin proxy that configures the session factory on first use."""

    def __call__(self, **kwargs):
        return get_session_factory()(**kwargs)

    def __getattr__(self, name):
        return getattr(get_session_factory(), name)


SessionLocal = _LazySessionFactory()


class SourceData(Base):
    __tablename__ = "sources"
    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    # optional serialized lists of entities and facts extracted during ingest
    entities = Column(Text, nullable=True)
    facts = Column(Text, nullable=True)

    datasets = relationship("Dataset", back_populates="source")


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    path = Column(String, nullable=False)
    content = Column(Text, nullable=True)

    source = relationship("SourceData", back_populates="datasets")


def init_db() -> None:
    """Create database tables."""

    Base.metadata.create_all(bind=get_engine())
