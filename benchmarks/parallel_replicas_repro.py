"""Minimal reproduction of the underlying ClickHouse parallel-replicas bug that
the fork works around (the workaround lives in
``ClickhouseServerAPI._create_table_as`` — split CTAS into a zero-row
structure-create + parallel INSERT; see tests_local/test_parallel_replicas.py).

This script exercises the RAW, un-worked-around pattern: a ``multiIf`` containing
both a constant-foldable ``isNull(...)`` (the analyzer folds it to
``_CAST(0, 'UInt8')``) and an ``arrayIntersect(...)``, materialized via
``CREATE TABLE ... AS`` under parallel replicas — which fails with
THERE_IS_NO_COLUMN because the ``_CAST`` constant's serialized name doesn't
round-trip across the coordinator<->replica boundary. Same root cause as
ClickHouse issues #74716 / #74367 / #74343 (all open).

Exit 0 if this ClickHouse runs the raw pattern (the upstream bug is fixed, so the
CTAS-split workaround is no longer needed), 1 if it still fails (workaround still
required). Re-run against new ClickHouse versions to know when to drop the split.

    CH_HOST=... CH_USER=default CH_PASS=... \
      uv run --package qwacked-splinkclickhouse python \
      packages/splink-clickhouse/benchmarks/parallel_replicas_repro.py
"""

from __future__ import annotations

import os
import sys

import clickhouse_connect

_PARALLEL = {
    "enable_parallel_replicas": 1,
    "max_parallel_replicas": 4,
    "cluster_for_parallel_replicas": "default",
    "parallel_replicas_for_non_replicated_merge_tree": 1,
}


def main() -> int:
    c = clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], port=int(os.environ.get("CH_PORT", "8443")),
        username=os.environ["CH_USER"], password=os.environ["CH_PASS"], secure=True,
    )
    print(f"ClickHouse {c.command('SELECT version()')}")
    c.command("DROP TABLE IF EXISTS pr_repro_src")
    c.command("DROP TABLE IF EXISTS pr_repro_out")
    c.command("CREATE TABLE pr_repro_src (id UInt64, emails Array(String)) "
              "ENGINE = MergeTree ORDER BY tuple()")
    c.command("INSERT INTO pr_repro_src SELECT number, [concat('e', toString(number % 7))] "
              "FROM numbers(3000)")
    sql = ("CREATE TABLE pr_repro_out ORDER BY tuple() AS SELECT id, "
           "multiIf(isNull(emails), -1, length(arrayIntersect(emails, emails)) >= 1, 1, 0) AS g "
           "FROM pr_repro_src")
    try:
        c.command(sql, settings=_PARALLEL)
        print("PASS: parallel replicas ran the Splink-shaped query — bug is fixed here.")
        rc = 0
    except Exception as e:
        print(f"FAIL: {str(e).split('DB::Exception:')[-1].strip()[:140]}")
        print("\n=> Blanket parallel replicas still unsupported for Splink on this server.")
        rc = 1
    finally:
        c.command("DROP TABLE IF EXISTS pr_repro_src")
        c.command("DROP TABLE IF EXISTS pr_repro_out")
    return rc


if __name__ == "__main__":
    sys.exit(main())
