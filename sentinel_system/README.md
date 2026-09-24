# Sentinel — Camera Registry (Module 1, M1.1)

The system of record for **what cameras exist and where**. This milestone is
the database layer only: models, schemas, repository, service, migrations
and tests.

Deliberately **not** in this milestone: API endpoints, AI/OCR, DeepStream,
Kafka, dashboard, watchlist, camera streaming. The registry holds no frames,
no detections and no analytics results — those belong to later modules and
would give this table a write rate it is not designed for.

---

## Layout

```
sentinel_system/
├── alembic.ini              # migration config; DB URL comes from the env
├── core/
│   ├── config.py            # settings from SENTINEL_* env vars
│   ├── database.py          # Base, engine, session; naming convention
│   └── exceptions.py        # SentinelError, the root of every domain error
├── registry/
│   ├── enums.py             # closed vocabularies
│   ├── models.py            # Camera (SQLAlchemy 2.x)
│   ├── schemas.py           # Pydantic v2 create/read/update/filter
│   ├── repository.py        # data access; SQL lives only here
│   ├── service.py           # use-cases, business rules, transactions
│   └── exceptions.py        # CameraNotFoundError, DuplicateCameraCodeError
└── migrations/              # Alembic environment + versions
```

Tests live in `tests/sentinel_registry/`, so the repository's existing
`./run test` picks them up with everything else.

## The layering, and why

```
service      use-cases. Owns business rules and the transaction boundary.
   │         Raises domain errors. Returns Pydantic objects, never ORM rows.
   ▼
repository   data access. Returns `Camera | None`, never raises domain
   │         errors, never commits.
   ▼
models       the table.
```

Three boundaries are load-bearing:

**The repository does not commit.** The caller owns the transaction, because
a later module will want to register a camera and write an audit row in one
transaction. A repository that commits on its own makes that impossible.

**The repository returns `None`; the service raises.** "No such camera" and
"that code is taken" are different outcomes that an API layer must map to
different statuses (404 vs 409). Returning `None` for both collapses them.

**The service returns `CameraRead`, not `Camera`.** A detached ORM instance
raises on attribute access once its session closes; handing one to a
serialiser is how `DetachedInstanceError` reaches a request log.

## What is an enum and what is a string

Enums are for sets the *code* reasons about — a scheduler checks
`status is CameraStatus.ACTIVE`, a stream reader switches on
`Protocol.RTSP`. Adding a member means new behaviour, so it belongs in a
migration and a review.

`department`, `owner`, `zone` and `district` are **not** enums. They are the
Gujarat Police organisational structure, which changes by administrative
order rather than software release — districts get reorganised, ranges get
renamed. A PostgreSQL ENUM would turn a clerical update into a schema
migration. They are validated, indexed strings.

`status` and `health` are separate vocabularies on purpose: an `active`
camera can be `unreachable`, and that combination — expected to work,
isn't — is precisely what operations needs to see.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SENTINEL_DATABASE_URL` | `sqlite+pysqlite:///./sentinel.db` | SQLAlchemy URL |
| `SENTINEL_SQL_ECHO` | `false` | Log every statement |
| `SENTINEL_DB_POOL_SIZE` | `5` | Pool size (ignored on SQLite) |
| `SENTINEL_DB_MAX_OVERFLOW` | `10` | Overflow connections |

PostgreSQL is the deployment target:

```bash
export SENTINEL_DATABASE_URL="postgresql+psycopg://sentinel:pw@localhost/sentinel"
```

The SQLite default exists so a fresh checkout can run the tests and the
example below without a database server. Every model is written to the
PostgreSQL feature set regardless.

## Migrations

```bash
cd sentinel_system
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
alembic downgrade base          # verified to reverse cleanly
alembic upgrade head --sql      # emit SQL for a DBA instead of applying
```

`core/database.py` sets a metadata **naming convention**, and it is the most
important line in this module. Without it the database invents names for
indexes and constraints, those names differ between PostgreSQL and SQLite
and between autogenerate runs, and a migration saying
`op.drop_constraint("ck_camera_a1b2c3")` is unrunnable anywhere but the
machine that produced it.

`compare_type` and `compare_server_default` are enabled in `env.py`, so a
changed column type or default is actually written into the migration
instead of being silently skipped.

## Usage

```python
from sentinel_system.core.database import session_scope
from sentinel_system.registry import (
    CameraCreate, CameraService, CameraType, Protocol, CameraStatus,
)

with session_scope() as session:
    service = CameraService(session)
    camera = service.create(CameraCreate(
        camera_code="AHM-SAT-0142",
        camera_name="Satellite Circle North",
        department="Gujarat Police",
        district="Ahmedabad",
        latitude=23.0225,
        longitude=72.5714,
        camera_type=CameraType.BULLET,
        protocol=Protocol.RTSP,
        stream_url="rtsp://10.20.4.11:554/Streaming/Channels/101",
        status=CameraStatus.ACTIVE,
    ))
    print(camera.id, camera.camera_code)
```

Partial updates distinguish "omitted" from "set to null":

```python
from sentinel_system.registry import CameraUpdate

service.update(camera.id, CameraUpdate(firmware_version="V6.0.0"))  # only that
service.update(camera.id, CameraUpdate(owner=None))                 # clears owner
service.update(camera.id, CameraUpdate())                           # no-op
```

## Validation

Enforced in the **schema** (Pydantic), so bad input never reaches the service:

- `camera_code` — upper-cased and trimmed, then `^[A-Z0-9][A-Z0-9_-]{1,62}[A-Z0-9]$`
- `latitude` −90..90, `longitude` −180..180
- `bearing_deg` 0 ≤ x < 360 (360 and 0 are the same heading)
- `resolution` normalised to `1920x1080` from `1920X1080` / `1920*1080`
- `stream_url` scheme must match `protocol` (RTSP + `https://` is rejected)
- `camera_type=ptz` requires `supports_ptz=true`
- unknown fields are **rejected**, not ignored — a typo'd field name that
  is silently dropped is a data-loss bug
- `order_by` is restricted to an allow-list, because it reaches `ORDER BY`

Enforced again in the **database** (CHECK constraints), because several
services, a migration and the occasional `psql` session share this table:
latitude/longitude ranges, `fps > 0`, `coverage_radius_m >= 0`, bearing
range, `length(camera_code) >= 3`, and `camera_code` UNIQUE.

Cross-field rules a payload cannot break on its own — `{"camera_type":"ptz"}`
against a stored row with `supports_ptz=false` — are checked in the service,
where the merged picture is visible.

## Design notes

**UUID primary key**, not a serial: camera records are created by several
district systems that must not coordinate over an ID sequence, and a
non-guessable identifier is the right default for something that will be
addressable over an API.

**Latitude/longitude as plain floats**, not PostGIS. The registry needs
"where is this camera" and bounding-box queries; a PostGIS dependency is a
deployment cost with no M1 payoff. The pair is indexed together so a bbox
scan stays cheap, and moving to `geography(Point)` later does not change
this module's public API.

**Capabilities are `NOT NULL DEFAULT false`.** A nullable boolean forces
every consumer to write `is True` to stay correct.

**Every timestamp is timezone-aware.** Gujarat is one timezone today, but
naive timestamps are a one-way door — once rows are stored without an
offset there is no recovering what they meant.

**Enums store their values, not their Python names.** Without
`values_callable`, SQLAlchemy persists `"ACTIVE"` while every payload and
log line says `"active"`; the mismatch stays invisible until someone writes
a raw query.

**Listing always applies a secondary sort on the primary key.** Rows tying
on the sort column would otherwise come back in planner order, which
differs between pages — so a row can appear on page 1 and again on page 2,
or be skipped.

**`delete` is a hard delete**, which is right while nothing references a
camera. Once detections point at cameras it should become a soft delete
(`status = decommissioned`), or the foreign keys will refuse.

## Tests

```bash
./run test tests/sentinel_registry -q     # in Docker
alpr/bin/python -m pytest tests/sentinel_registry -q   # native
```

143 tests: enums, schema validation, model defaults and constraints,
repository queries/filtering/pagination, service use-cases and domain
errors.

Each test gets its own in-memory SQLite database built from the same
declarative metadata the migrations are generated from. SQLite is not the
deployment target, so anything genuinely needing PostgreSQL (native ENUM
types, concurrent inserts racing the unique index) is noted where it is
tested rather than assumed.

## Next (not in M1.1)

M1.2 — FastAPI routers over `CameraService`. `core/database.get_session` is
already shaped as a generator so it can be used directly as a dependency,
and every domain error already carries a stable `code` for an error
envelope.
