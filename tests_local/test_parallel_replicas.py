"""The ``max_parallel_replicas`` kwarg: it applies the session settings, and
running the full pipeline across replicas yields the identical clustering to
single-replica — parallelism must not change the answer.

Server-gated via the ``ch_client`` fixture (see conftest)."""

from __future__ import annotations

from _data import full_settings, link, partition, planted_clusters

from splinkclickhouse import ClickhouseServerAPI


def test_kwarg_sets_parallel_session_settings(ch_client):
    ClickhouseServerAPI(ch_client, max_parallel_replicas=4)
    assert int(ch_client.command("SELECT getSetting('enable_parallel_replicas')")) == 1
    assert int(ch_client.command("SELECT getSetting('max_parallel_replicas')")) == 4


def test_default_leaves_parallel_replicas_off(ch_client):
    ClickhouseServerAPI(ch_client)
    assert int(ch_client.command("SELECT getSetting('enable_parallel_replicas')")) == 0


def test_parallel_clustering_matches_single_replica(ch_client):
    df = planted_clusters()
    settings = full_settings()

    _, single = link(ClickhouseServerAPI(ch_client), df, settings)
    _, parallel = link(ClickhouseServerAPI(ch_client, max_parallel_replicas=4), df, settings)

    assert partition(parallel) == partition(single)
