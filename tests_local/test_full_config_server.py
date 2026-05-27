"""The full identity-resolve comparison shape against a real ClickHouse server:
JaroWinkler+TF, ArrayIntersect (the array column chDB couldn't represent),
geo-distance, EM training, predict, and connected-components clustering.

Server-gated via the ``ch_client`` fixture (see conftest)."""

from __future__ import annotations

from _data import full_settings, link, partition, planted_clusters, planted_partition

from splinkclickhouse import ClickhouseServerAPI


def test_full_config_with_arrays_and_clustering(ch_client):
    df = planted_clusters()
    predictions, clusters = link(ClickhouseServerAPI(ch_client), df, full_settings())

    # Every comparison produced a gamma column (its SQL evaluated, incl.
    # arrayIntersect), and JaroWinkler discriminated rather than returning a
    # constant level.
    for col in ("gamma_first_name", "gamma_emails", "gamma_latitude_longitude"):
        assert col in predictions.columns, col
    assert predictions["gamma_first_name"].nunique() > 1

    # Surname blocking makes cross-entity pairs impossible, and the strong
    # email/geo signal merges every duplicate => the recovered clustering is
    # exactly the planted one.
    assert partition(clusters) == planted_partition(df)
