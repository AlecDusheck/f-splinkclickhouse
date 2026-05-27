"""Server-backed fixtures. Connects to the ClickHouse given by CH_HOST / CH_USER
/ CH_PASS (CH_PORT defaults to 8443 + TLS, i.e. ClickHouse Cloud; set CH_PORT=8123
for a local server). Skips cleanly when no server is reachable."""

from __future__ import annotations

import os
import uuid

import pytest


def _client(database: str | None = None):
    import clickhouse_connect

    port = int(os.environ.get("CH_PORT", "8443"))
    return clickhouse_connect.get_client(
        host=os.environ.get("CH_HOST", "localhost"),
        port=port,
        username=os.environ.get("CH_USER", "default"),
        password=os.environ.get("CH_PASS", ""),
        secure=port == 8443,
        database=database,
        # Local single-binary builds segfault JIT-compiling; harmless on Cloud.
        settings={"compile_expressions": 0, "compile_aggregate_expressions": 0},
    )


@pytest.fixture
def ch_client():
    pytest.importorskip("clickhouse_connect")
    db = f"splink_fork_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = _client()
        admin.command(f"CREATE DATABASE IF NOT EXISTS {db}")
    except Exception as e:  # noqa: BLE001 — no server => skip (set CH_HOST/USER/PASS)
        pytest.skip(f"no ClickHouse server: {type(e).__name__}")
    client = _client(database=db)
    yield client
    client.close()
    admin.command(f"DROP DATABASE IF EXISTS {db}")
    admin.close()
