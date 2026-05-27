"""How Splink's blocking-join workload scales across ClickHouse compute replicas.

Generates synthetic person rows server-side (nothing touches the client),
then times the Splink-shaped heavy query — a self-join on a block key plus
per-pair JaroWinkler/geoDistance scoring — while sweeping ``max_parallel_replicas``.
Reports wall-clock speedup and, from ``ProfileEvents``, how many replicas were
actually available and used (``system.clusters`` can advertise more than are live).

This measures the WORKLOAD SHAPE on hand-written SQL (which scales ~linearly).
Splink's own generated SQL currently can't run across replicas on CH 25.12 — see
``tests_local/test_parallel_replicas.py`` for that (separate) blocker.

    CH_HOST=... CH_USER=default CH_PASS=... \
      uv run --package qwacked-splinkclickhouse python \
      packages/splink-clickhouse/benchmarks/parallel_replicas.py --rows 10_000_000
"""

from __future__ import annotations

import argparse
import os
import time

import clickhouse_connect

_FIRST = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi",
          "ivan", "judy", "mallory", "niaj", "olivia", "peggy", "sybil", "trent"]
_CLUSTER = "default"


def connect() -> clickhouse_connect.driver.client.Client:
    return clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], port=int(os.environ.get("CH_PORT", "8443")),
        username=os.environ["CH_USER"], password=os.environ["CH_PASS"], secure=True,
    )


def generate(client, table: str, rows: int, surnames: int) -> int:
    """Build the synthetic table, ordered on the block key so it both seeks and
    splits cleanly across replicas. Returns the candidate-pair count."""
    names = "[" + ",".join(f"'{n}'" for n in _FIRST) + "]"
    client.command(f"DROP TABLE IF EXISTS {table}")
    client.command(f"""
        CREATE TABLE {table} (
            unique_id UInt64, first_name String, surname String,
            dob Date, lat Float64, lon Float64
        ) ENGINE = MergeTree ORDER BY (surname, unique_id)
    """)
    client.command(f"""
        INSERT INTO {table} SELECT
            number,
            {names}[1 + cityHash64(number, 'f') % {len(_FIRST)}],
            concat('sur', toString(cityHash64(number, 's') % {surnames})),
            toDate('1950-01-01') + cityHash64(number, 'd') % 25000,
            51.5 + cityHash64(number, 'la') % 100000 / 100000.0,
            -0.5 + cityHash64(number, 'lo') % 100000 / 100000.0
        FROM numbers_mt({rows})
    """)
    return client.command(
        f"SELECT toUInt64(sum(c * (c - 1) / 2)) "
        f"FROM (SELECT count() c FROM {table} GROUP BY surname)"
    )


def _heavy_sql(table: str) -> str:
    return f"""
        SELECT count(),
               avg(jaroWinklerSimilarity(a.first_name, b.first_name)),
               avg(geoDistance(a.lon, a.lat, b.lon, b.lat))
        FROM {table} AS a INNER JOIN {table} AS b ON a.surname = b.surname
        WHERE a.unique_id < b.unique_id
    """


def time_mode(client, table: str, replicas: int, repeats: int) -> tuple[float, int, int]:
    settings = {
        "enable_parallel_replicas": int(replicas > 1),
        "max_parallel_replicas": replicas,
        "cluster_for_parallel_replicas": _CLUSTER,
        "parallel_replicas_for_non_replicated_merge_tree": 1,
    }
    best, qid = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        qid = client.query(_heavy_sql(table), settings=settings).query_id
        best = min(best, time.perf_counter() - t0)
    client.command("SYSTEM FLUSH LOGS")
    time.sleep(2)
    avail, used = client.query(
        "SELECT ProfileEvents['ParallelReplicasAvailableCount'], "
        "ProfileEvents['ParallelReplicasUsedCount'] "
        f"FROM system.query_log WHERE query_id = '{qid}' AND type = 'QueryFinish' LIMIT 1"
    ).result_rows[0]
    return best, avail, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000_000)
    ap.add_argument("--surnames", type=int, default=100_000,
                    help="block-key cardinality; lower => bigger blocks => more pairs")
    ap.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--table", default="splink_parallel_bench")
    args = ap.parse_args()

    client = connect()
    print(f"server {client.command('SELECT version()')}, generating {args.rows:,} rows...")
    pairs = generate(client, args.table, args.rows, args.surnames)
    print(f"{args.rows:,} rows, ~{pairs:,} candidate pairs\n")

    print(f"{'replicas':>8}  {'wall':>7}  {'speedup':>7}  available  used")
    base = None
    for n in args.replicas:
        wall, avail, used = time_mode(client, args.table, n, args.repeats)
        base = base or wall
        print(f"{n:>8}  {wall:6.2f}s  {base / wall:6.2f}x  {avail:>9}  {used}")

    client.command(f"DROP TABLE IF EXISTS {args.table}")


if __name__ == "__main__":
    main()
