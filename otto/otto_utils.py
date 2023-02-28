from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
from scipy.sparse import csr_matrix


pd.options.display.float_format = '{:,.4f}'.format

DATA_FOLDER = Path("/data/otto/")
DATA_FOLDER = DATA_FOLDER if DATA_FOLDER.exists() else Path("/home/sirpantene/data/otto/")

TRAIN_PROCESSED = DATA_FOLDER / "train_parquet"
TRAIN_PROCESSED.mkdir(parents=True, exist_ok=True)

TEST_PROCESSED = DATA_FOLDER / "test_parquet"
TEST_PROCESSED.mkdir(parents=True, exist_ok=True)

VALIDATION_PATH = DATA_FOLDER / "validation"
VALIDATION_PATH.mkdir(parents=True, exist_ok=True)


def make_sparse_matrix(df: pl.DataFrame):
    row = df["session"].to_numpy()
    col = df["aid"].to_numpy()
    weight = df["weight"].to_numpy().astype(np.float32)
    
    return csr_matrix((weight, (row, col)))


def dataset_action_weights(df):
    df_action_weights = pl.DataFrame({
        "type": [0, 1, 2], "weight": [10, 30, 60]
    })

    return (
        df
        .join(df_action_weights, on="type", how="inner")
        .groupby(["session", "aid"])
        .agg([
            pl.sum("weight")
        ])
    )


def dataset_clicks_only(df):
    return (
        df
        .filter(pl.col("type") == 0)
        .groupby(["session", "aid"])
        .agg([
            pl.lit(1).alias("weight")
        ])
    )

def data_stats(df):
    
    pldf_user_type_stats = (
        df
        .groupby(["session", "type"])
        .agg([
            pl.n_unique("aid").alias("uniq_aids"),
        ])
        .sort("uniq_aids", reverse=True)
        .pivot(
            values="uniq_aids", index="session", columns="type"
        )
        .rename({
            "0": "uniq_clicks",
            "1": "uniq_carts",
            "2": "uniq_orders",
        })
        .fill_null(0)
    )
    
    pldf_user_stats = (
        df
        .groupby("session")
        .agg([
            pl.min("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("min_ts"),
            pl.max("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("max_ts"),
            pl.count("aid").alias("session_total_count"),
            pl.n_unique("aid").alias("uniq_aids"),
            pl.mean("type").alias("session_buys_rate"),
        ])
        .sort("uniq_aids", reverse=True)
        .join(pldf_user_type_stats, on="session", how="inner")
    )
    
    pldf_item_type_stats = (
        df
        .groupby(["aid", "type"])
        .agg([
            pl.n_unique("session").alias("uniq_sessions"),
        ])
        .sort("uniq_sessions", reverse=True)
        .pivot(
            values="uniq_sessions", index="aid", columns="type"
        )
        .rename({
            "0": "uniq_clicks",
            "1": "uniq_carts",
            "2": "uniq_orders",
        })
        .fill_null(0)
    )

    pldf_item_stats = (
        df
        .groupby("aid")
        .agg([
            pl.min("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("min_ts"),
            pl.max("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("max_ts"),
            pl.count("session").alias("aid_total_count"),
            pl.n_unique("session").alias("uniq_sessions"),
            pl.mean("type").alias("aid_buys_rate"),
        ])
        .sort("uniq_sessions", reverse=True)
        .join(pldf_item_type_stats, on="aid", how="inner")
    )

    return pldf_user_stats, pldf_item_stats



def data_stats(df):
    
    pldf_user_type_stats = (
        df
        .groupby(["session", "type"])
        .agg([
            pl.n_unique("aid").alias("uniq_aids"),
        ])
        .sort("uniq_aids", reverse=True)
        .pivot(
            values="uniq_aids", index="session", columns="type"
        )
        .rename({
            "0": "uniq_clicks",
            "1": "uniq_carts",
            "2": "uniq_orders",
        })
        .fill_null(0)
    )
    
    pldf_user_stats = (
        df
        .groupby("session")
        .agg([
            pl.min("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("min_ts"),
            pl.max("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("max_ts"),
            pl.count("aid").alias("session_total_count"),
            pl.n_unique("aid").alias("uniq_aids"),
            pl.mean("type").alias("session_buys_rate"),
        ])
        .sort("uniq_aids", reverse=True)
        .join(pldf_user_type_stats, on="session", how="inner")
    )
    
    pldf_item_type_stats = (
        df
        .groupby(["aid", "type"])
        .agg([
            pl.n_unique("session").alias("uniq_sessions"),
        ])
        .sort("uniq_sessions", reverse=True)
        .pivot(
            values="uniq_sessions", index="aid", columns="type"
        )
        .rename({
            "0": "uniq_clicks",
            "1": "uniq_carts",
            "2": "uniq_orders",
        })
        .fill_null(0)
    )

    pldf_item_stats = (
        df
        .groupby("aid")
        .agg([
            pl.min("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("min_ts"),
            pl.max("ts").cast(
                pl.Datetime(time_unit="ms", time_zone="Etc/GMT-2")
            ).alias("max_ts"),
            pl.count("session").alias("aid_total_count"),
            pl.n_unique("session").alias("uniq_sessions"),
            pl.mean("type").alias("aid_buys_rate"),
        ])
        .sort("uniq_sessions", reverse=True)
        .join(pldf_item_type_stats, on="aid", how="inner")
    )

    return pldf_user_stats, pldf_item_stats


def calc_valid_score(df_submission: pl.DataFrame, topk: int = 20):
    submission = df_submission.to_pandas()
    submission['session'] = submission.session_type.apply(lambda x: int(x.split('_')[0]))
    submission['type'] = submission.session_type.apply(lambda x: x.split('_')[1])
    submission.labels = submission.labels.apply(lambda x: [int(i) for i in x.split(' ')[:topk]])
#     submission.labels = submission.labels.apply(lambda x: [int(i) for i in x.split(' ')])
    
    val_df_valid_input = pl.read_parquet(VALIDATION_PATH / "valid.parquet", use_pyarrow=True)
    val_df_valid_targets = pl.read_parquet(VALIDATION_PATH / "test_labels.parquet", use_pyarrow=True)
    test_labels = (
        val_df_valid_targets
        .join(val_df_valid_input.select(["session"]).unique(), on="session", how="inner")
        .to_pandas()
    )

    test_labels = submission.merge(test_labels, how='outer', on=['session', 'type'])
    labels_null_idx = test_labels["ground_truth"].isnull()
    test_labels["ground_truth"].loc[labels_null_idx] = (
        test_labels["ground_truth"].loc[labels_null_idx]
        .apply(lambda x: [])
    )
    labels_null_idx = test_labels["labels"].isnull()
    test_labels["labels"].loc[labels_null_idx] = (
        test_labels["labels"].loc[labels_null_idx]
        .apply(lambda x: [])
    )
    test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
    # test_labels['hits'] = test_labels.hits.clip(0, topk) # when hits greater than clipped gt_count
    test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0, topk)
    # test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0, 20)
    
    recall_per_type = (
        test_labels.groupby(['type'])['hits'].sum() / 
        test_labels.groupby(['type'])['gt_count'].sum() 
    )

    action_weights = pd.Series({
        'clicks': 0.10, 
        'carts': 0.30, 
        'orders': 0.60
    })

    score = (recall_per_type * action_weights).sum()
    print(f"validation score: {score}")
    print(f"recall per type: {recall_per_type}")
    return test_labels
