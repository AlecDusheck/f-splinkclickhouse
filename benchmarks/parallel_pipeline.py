"""End-to-end Splink pipeline benchmark: parallel replicas off vs on.

Generates a synthetic person table server-side (nothing big on the client), then
runs the full pipeline — EM training, predict, connected-components clustering —
under each config, reporting total wall-clock and PEAK single-node memory.

    CH_HOST=... CH_USER=default CH_PASS=... \
      uv run --package qwacked-splinkclickhouse python \
      packages/splink-clickhouse/benchmarks/parallel_pipeline.py --rows 2_000_000
"""

from __future__ import annotations

import argparse
import os
import time
import uuid

import clickhouse_connect
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, block_on

import splinkclickhouse.comparison_library as cl_ch
from splinkclickhouse import ClickhouseServerAPI

_FIRST = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi",
          "ivan", "judy", "mallory", "niaj", "olivia", "peggy", "sybil", "trent"]


def _client(database=None, log_comment=None):
    settings = {"log_comment": log_comment} if log_comment else {}
    return clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], port=int(os.environ.get("CH_PORT", "8443")),
        username=os.environ["CH_USER"], password=os.environ["CH_PASS"], secure=True,
        database=database, settings=settings,
    )


def generate(client, table: str, rows: int, surnames: int) -> int:
    names = "[" + ",".join(f"'{n}'" for n in _FIRST) + "]"
    client.command(f"DROP TABLE IF EXISTS {table}")
    client.command(f"""
        CREATE TABLE {table} (
            unique_id UInt64, first_name String, surname String,
            emails Array(String), latitude Float64, longitude Float64
        ) ENGINE = MergeTree ORDER BY (surname, unique_id)
    """)
    client.command(f"""
        INSERT INTO {table} SELECT
            number,
            {names}[1 + cityHash64(number, 'f') % {len(_FIRST)}],
            concat('sur', toString(cityHash64(number, 's') % {surnames})),
            [concat('e', toString(cityHash64(number, 'e') % {rows // 2}))],
            51.5 + cityHash64(number, 'la') % 100000 / 100000.0,
            -0.1 + cityHash64(number, 'lo') % 100000 / 100000.0
        FROM numbers_mt({rows})
    """)
    return client.command(
        f"SELECT toUInt64(sum(c * (c - 1) / 2)) "
        f"FROM (SELECT count() c FROM {table} GROUP BY surname)"
    )


def _settings() -> SettingsCreator:
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.JaroWinklerAtThresholds("first_name", [0.9]).configure(
                term_frequency_adjustments=True
            ),
            cl.ArrayIntersectAtSizes("emails", [1]),
            cl_ch.DistanceInKMAtThresholds("latitude", "longitude", [1, 10]),
        ],
        blocking_rules_to_generate_predictions=[block_on("surname")],
    )


def run_config(db: str, table: str, *, replicas: int) -> dict:
    tag = f"bench_{uuid.uuid4().hex[:10]}"
    client = _client(database=db, log_comment=tag)
    api = ClickhouseServerAPI(client, max_parallel_replicas=replicas)
    linker = Linker(table, _settings(), db_api=api)

    t = time.perf_counter()
    linker.training.estimate_probability_two_random_records_match(
        [block_on("surname")], recall=0.9
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("surname"))
    predictions = linker.inference.predict()
    linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=0.5
    )
    wall = time.perf_counter() - t

    client.command("SYSTEM FLUSH LOGS")
    time.sleep(2)
    # peak single-node memory across ALL compute nodes (incl. parallel replicas)
    peak_mem, read_rows = client.query(
        "SELECT max(memory_usage), sum(read_rows) "
        "FROM clusterAllReplicas('default', system.query_log) "
        f"WHERE log_comment = '{tag}' AND type = 'QueryFinish'"
    ).result_rows[0]
    client.close()
    return {"wall": wall, "peak_mem": peak_mem, "read_rows": read_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--surnames", type=int, default=20_000)
    ap.add_argument("--replicas", type=int, default=4)
    args = ap.parse_args()

    db = f"bench_{uuid.uuid4().hex[:8]}"
    admin = _client()
    admin.command(f"CREATE DATABASE {db}")
    table = "people"  # unqualified: each config's client is connected to `db`
    try:
        print(f"server {admin.command('SELECT version()')}, generating {args.rows:,} rows...")
        pairs = generate(admin, f"{db}.{table}", args.rows, args.surnames)
        print(f"{args.rows:,} rows, ~{pairs:,} candidate pairs\n")

        configs = [
            ("baseline (1 replica)", dict(replicas=1)),
            ("parallel replicas", dict(replicas=args.replicas)),
        ]
        results = {}
        for name, kw in configs:
            print(f"running {name}...")
            try:
                results[name] = run_config(db, table, **kw)
            except Exception as e:  # noqa: BLE001 — a config OOMing is itself a result
                msg = str(e).split("DB::Exception:")[-1].strip()[:80]
                results[name] = {"error": msg}
                print(f"  {name} FAILED: {msg}")

        base = results["baseline (1 replica)"]
        print(f"\n{'config':>20}  {'wall':>8}  {'peak mem':>10}  {'mem vs base':>11}  {'speedup':>7}")
        for name, _ in configs:
            r = results[name]
            if "error" in r:
                print(f"{name:>20}  {'FAILED':>8}  {r['error']}")
                continue
            mem_gib = r["peak_mem"] / 1024**3
            mem_x = f"{base['peak_mem'] / r['peak_mem']:.2f}x" if "peak_mem" in base else "-"
            wall_x = f"{base['wall'] / r['wall']:.2f}x" if "wall" in base else "-"
            print(f"{name:>20}  {r['wall']:7.1f}s  {mem_gib:8.2f}GiB  {mem_x:>11}  {wall_x:>7}")
    finally:
        admin.command(f"DROP DATABASE IF EXISTS {db}")


if __name__ == "__main__":
    main()
