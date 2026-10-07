-- Plain-Postgres part of the pytweezer schema. Safe to re-run.
-- The TimescaleDB part (hypertable + compression) is timescale.sql.

-- One row per field per reading: a narrow table, so any logger can add
-- measurements and fields without a migration.
CREATE TABLE IF NOT EXISTS readings (
    time        timestamptz      NOT NULL,
    measurement text             NOT NULL,
    field       text             NOT NULL,
    value       double precision NOT NULL,
    tags        jsonb            NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS readings_series_idx
    ON readings (measurement, field, time DESC);

-- One row per Experiment Manager run, upserted when it starts and finishes.
CREATE TABLE IF NOT EXISTS runs (
    rid          integer PRIMARY KEY,
    experiment   text,
    class_name   text,
    label        text,
    submitter    text,
    arguments    jsonb,
    scan         jsonb,
    status       text,
    error        text,
    t_submit     timestamptz,
    t_start      timestamptz,
    t_end        timestamptz,
    points_done  integer,
    points_total integer,
    h5_path      text,
    simulated    boolean
);

-- One row per scan point that ran. scan_values holds the scanned arguments,
-- scalars the 0-d numeric results recorded in that point.
CREATE TABLE IF NOT EXISTS points (
    rid         integer     NOT NULL,
    point_index integer     NOT NULL,
    t_start     timestamptz,
    t_end       timestamptz,
    scan_values jsonb       NOT NULL DEFAULT '{}',
    scalars     jsonb       NOT NULL DEFAULT '{}',
    PRIMARY KEY (rid, point_index)
);
CREATE INDEX IF NOT EXISTS points_t_start_idx ON points (t_start);
