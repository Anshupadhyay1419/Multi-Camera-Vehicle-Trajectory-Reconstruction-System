"""Alembic environment for the Sentinel platform.

Two things here are load-bearing for migrations that stay reviewable:

`target_metadata` points at the same declarative Base the models use, so
`--autogenerate` diffs the real schema rather than a hand-maintained copy.

`compare_type` and `compare_server_default` are on. Without them Alembic
notices added and dropped columns but silently ignores a column whose type
or default changed, which is exactly the migration you most want written
for you.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import every model module before reading metadata: a model that is not
# imported is not in Base.metadata, and autogenerate would cheerfully
# generate a migration that DROPS its table.
from sentinel_system.core.database import Base
from sentinel_system.registry import models as _registry_models  # noqa: F401
from sentinel_system.verification import models as _verification_models  # noqa: F401
from sentinel_system.bulk_import import models as _bulk_import_models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """The URL to migrate, from the environment, not from alembic.ini."""
    return os.environ.get(
        "SENTINEL_DATABASE_URL", "sqlite+pysqlite:///./sentinel.db"
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (`alembic upgrade --sql`).

    This is how a migration gets handed to a DBA who will apply it during a
    maintenance window rather than letting the application run it.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # SQLite cannot ALTER most things in place; batch mode rewrites
            # the table instead. A no-op on PostgreSQL, so it is safe to
            # leave on and means one migration script runs on both.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
