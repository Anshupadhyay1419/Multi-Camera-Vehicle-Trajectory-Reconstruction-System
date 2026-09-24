"""Declarative base, engine and session plumbing.

The naming convention on the metadata is the single most important thing
in this file. Without it, SQLAlchemy lets the database invent names for
indexes, unique constraints and check constraints -- and those invented
names differ between PostgreSQL and SQLite, and between one Alembic
autogenerate run and the next. A migration that says
`op.drop_constraint("ck_camera_a1b2c3")` is then unrunnable anywhere but
the machine that generated it. Fixing the convention up front is what makes
`alembic revision --autogenerate` produce diffs that are stable, reviewable
and reversible.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sentinel_system.core.config import get_settings

# %(constraint_name)s is supplied by us for check constraints; everything
# else is derived from the table and columns involved, so a rename shows up
# as a rename instead of a drop-and-create of a differently-named object.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by every model in the platform."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def build_engine(url: str | None = None, **kwargs: object) -> Engine:
    """Create an Engine for `url`, defaulting to the configured database."""
    settings = get_settings()
    target = url or settings.database_url
    options: dict[str, object] = {"echo": settings.sql_echo, "future": True}

    if target.startswith("sqlite"):
        # SQLite has no server-side pool to size, and the default pool
        # would hand the same connection to threads that must not share it.
        from sqlalchemy.pool import StaticPool

        options["connect_args"] = {"check_same_thread": False}
        if ":memory:" in target:
            options["poolclass"] = StaticPool
    else:
        options["pool_size"] = settings.db_pool_size
        options["max_overflow"] = settings.db_max_overflow
        options["pool_pre_ping"] = True

    options.update(kwargs)
    return create_engine(target, **options)  # type: ignore[arg-type]


def get_engine() -> Engine:
    """The process-wide engine, built on first use."""
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """The process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False
        )
    return _session_factory


def get_session() -> Iterator[Session]:
    """Yield a Session, closing it afterwards.

    Shaped as a generator so it can be used directly as a FastAPI
    dependency in M1.2 without a wrapper.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional scope: commit on success, roll back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def configure(url: str, **kwargs: object) -> Engine:
    """Bind this module to a specific database, replacing any current engine.

    The integration point for a host application. The ALPR API already
    resolves which database the deployment uses (config.yaml `database.path`,
    `$ALPR_DB_PATH`, `$DB_URL`), and the registry must land in that same
    database rather than quietly opening a second one beside it -- two
    database files, one of which nobody backs up, is how a camera registry
    goes missing.

    Deliberately explicit rather than having this module import the ALPR
    config itself: the dependency belongs to the host, which knows it has
    both, not to `sentinel_system`, which should stay usable on its own.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = build_engine(url, **kwargs)
    _session_factory = None
    return _engine


def reset_engine() -> None:
    """Drop the cached engine and factory. Used by tests."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
