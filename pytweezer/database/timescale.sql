-- TimescaleDB part of the pytweezer schema, applied after schema.sql when the
-- timescaledb extension is installed. Safe to re-run.

SELECT create_hypertable(
    'readings', by_range('time', INTERVAL '7 days'),
    if_not_exists => TRUE, migrate_data => TRUE
);

DO $$
BEGIN
    IF NOT (SELECT compression_enabled FROM timescaledb_information.hypertables
            WHERE hypertable_schema = current_schema()
              AND hypertable_name = 'readings') THEN
        ALTER TABLE readings SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'measurement, field',
            timescaledb.compress_orderby = 'time DESC'
        );
    END IF;
END $$;

SELECT add_compression_policy('readings', INTERVAL '7 days', if_not_exists => TRUE);
