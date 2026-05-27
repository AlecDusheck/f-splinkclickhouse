"""Synthetic data + linkage helpers shared by the server-backed tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, block_on

import splinkclickhouse.comparison_library as cl_ch

_FIRST = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]


def planted_clusters(n_entities: int = 60, seed: int = 42) -> pd.DataFrame:
    """One unique surname per entity (the block key); 1–3 records each sharing an
    email + near-identical lat/lon, with a first-name typo on duplicates so the
    probabilistic comparisons have work to do."""
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for e in range(n_entities):
        first = str(rng.choice(_FIRST))
        lat0, lon0 = 51.5 + e * 0.01, -0.1 + e * 0.01
        for d in range(int(rng.integers(1, 4))):
            fn = first if d == 0 else first[:-1] + str(rng.choice(list("xyz")))
            rows.append({
                "unique_id": uid,
                "first_name": fn,
                "surname": f"sur{e:03d}",
                # python list -> the server API types it Array(String)
                "emails": [f"user{e}@example.com"] + ([f"alt{e}@x.com"] if d % 2 else []),
                "latitude": float(lat0 + rng.normal(0, 0.0003)),
                "longitude": float(lon0 + rng.normal(0, 0.0003)),
                "true_entity": e,
            })
            uid += 1
    return pd.DataFrame(rows)


def full_settings() -> SettingsCreator:
    """The full comparison shape identity-resolve relies on: JaroWinkler+TF,
    ArrayIntersect (array column), and the ClickHouse geo-distance comparison."""
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


def link(db_api, df, settings, threshold: float = 0.5):
    """Run the full path — train, predict, cluster — returning (predictions, clusters)."""
    linker = Linker(df, settings, db_api=db_api)
    linker.training.estimate_probability_two_random_records_match(
        [block_on("surname")], recall=0.9
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=1e5)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("surname"))
    predictions = linker.inference.predict()
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=threshold
    )
    return predictions.as_pandas_dataframe(), clusters.as_pandas_dataframe()


def partition(clusters: pd.DataFrame) -> set[frozenset]:
    """The clustering as a set of member-id groups (label-independent)."""
    return {
        frozenset(g) for g in clusters.groupby("cluster_id")["unique_id"].apply(list)
    }


def planted_partition(df: pd.DataFrame) -> set[frozenset]:
    return {frozenset(g) for g in df.groupby("true_entity")["unique_id"].apply(list)}
