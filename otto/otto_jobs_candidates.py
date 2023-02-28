from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from implicit.nearest_neighbours import CosineRecommender
from implicit.nearest_neighbours import TFIDFRecommender
from tqdm import tqdm

from notebooks.otto.otto_utils import VALIDATION_PATH, TRAIN_PROCESSED, TEST_PROCESSED
from notebooks.otto.otto_candidates_covisit import CovisitationRecommender
from notebooks.otto.otto_implicit import (
    make_sparse_matrix,
    implicit_old_weight_interactions,
    implicit_new_weight_interactions,
    implicit_batch_candidates_for_all_types,
)


def print_candidates_hit_rate(df_candidates):
    df_sessions_with_positives = (
        df_candidates
        .groupby(["session"]).agg(pl.sum("target"))
        .filter(pl.col("target") > 0)
        .select(["session"])
    )

    print("sessions with positives in candidates: ", df_sessions_with_positives["session"].n_unique())
    print("sessions with positives in test dataframe: ", df_candidates["session"].n_unique())
    print(
        "hit rate: ", 
        df_sessions_with_positives["session"].n_unique() / df_candidates["session"].n_unique()
    )
    
    df_candidates_with_positives = df_candidates.join(df_sessions_with_positives, on="session", how="inner")
    print(df_candidates_with_positives["target"].value_counts())
    
    return df_candidates_with_positives


def generate_candidates_covisit_all(
    _dataset_type: str,
    save_path: Path = Path("reranker_finale"),
    topk: int = 200,
):
    df_test, df_target = None, None
    carts_orders, buys2buys, clicks = None, None, None
    df = None
    print("Reading dataset...: ", _dataset_type)
    if _dataset_type == "valid":
        df_test = pl.read_parquet(VALIDATION_PATH / "valid.parquet", use_pyarrow=True)
        df_target = pl.read_parquet(VALIDATION_PATH / "test_labels.parquet", use_pyarrow=True)
        carts_orders = pl.read_parquet(save_path / "__valid__covisit_carts_orders_all_v3.parquet")
        buys2buys = pl.read_parquet(save_path / "__valid__covisit_buys2buys_all_v4.parquet")
        clicks = pl.read_parquet(save_path / "__valid__covisit_clicks_all_v3.parquet")

    elif _dataset_type == "subm":
        # read test data
        df_test = pl.read_parquet(TEST_PROCESSED, use_pyarrow=True)
        carts_orders = pl.read_parquet(save_path / "__subm__covisit_carts_orders_all_v1.parquet")
        buys2buys = pl.read_parquet(save_path / "__subm__covisit_buys2buys_all_v1.parquet")
        clicks = pl.read_parquet(save_path / "__subm__covisit_clicks_all_v1.parquet")
    elif _dataset_type == "subm_no_fw":
        # read test data - cutted first week
        pass
        df_test = pl.read_parquet(TEST_PROCESSED, use_pyarrow=True)
        # carts_orders = pl.read_parquet(save_path / "__valid__covisit_carts_orders_all_v3.parquet")
        # buys2buys = pl.read_parquet(save_path / "__valid__covisit_buys2buys_all_v4.parquet")
        # clicks = pl.read_parquet(save_path / "__valid__covisit_clicks_all_v3.parquet")
    else:
        raise ValueError(f"Wrong _dataset_type {_dataset_type}")

    # Use top X for clicks, carts and orders
    clicks_th = 15
    carts_th  = 20
    orders_th = 20
    TOPK_RECOMMEND = 20
    TOPK_RERANK = 40

    def get_top(df, th):
        return (
            df
            .with_column(pl.lit(1).alias("ones"))
            .with_column(pl.col("ones").cumsum().over("aid").alias("rank"))
            .filter(pl.col("rank") <= th)
        )

    carts_orders_top = get_top(carts_orders, carts_th)
    buys2buys_top = get_top(buys2buys, orders_th)
    clicks_top = get_top(clicks, clicks_th)

    df = df_test.unique().sort(["session", "ts"])
    top_clicks = df.filter(pl.col("type") == 0)["aid"].value_counts(sort=True)[:TOPK_RECOMMEND]["aid"].to_list()
    top_carts = df.filter(pl.col("type") == 1)["aid"].value_counts(sort=True)[:TOPK_RECOMMEND]["aid"].to_list()
    top_orders = df.filter(pl.col("type") == 2)["aid"].value_counts(sort=True)[:TOPK_RECOMMEND]["aid"].to_list()

    covisit_rec = CovisitationRecommender(
        df_top_k_buys=carts_orders_top,
        df_top_k_buy2buy=buys2buys_top,
        df_top_k_clicks=clicks_top,
        top_carts=top_carts,
        top_orders=top_orders,
        top_clicks=top_clicks,
    )

    candidates_dict = {"session": [], "type": [], "candidates": [], "rank": []}
    types = ["clicks", "carts", "orders"]

    test_sessions_dict = df.groupby('session').agg([pl.list("aid"), pl.list("type")])
    test_sessions_dict = dict(zip(
        test_sessions_dict["session"].to_list(),
        tuple(zip(test_sessions_dict["aid"].to_list(), test_sessions_dict["type"].to_list()))
    ))

    for session_id, (session_aid_list, session_type_list) in tqdm(test_sessions_dict.items()):
        rec_items_clicks = covisit_rec.recommend_clicks(session_aid_list, session_type_list, topk)
        rec_items_carts = covisit_rec.recommend_carts(session_aid_list, session_type_list, topk)
        rec_items_buys = covisit_rec.recommend_buys(session_aid_list, session_type_list, topk)

        candidates = [rec_items_clicks, rec_items_carts, rec_items_buys]
        ranks = [
            np.arange(1, len(rec_items) + 1).tolist()
            for rec_items in candidates
        ]
        
        candidates_dict["session"].extend([session_id] * len(types))
        candidates_dict["type"].extend(types)
        candidates_dict["candidates"].extend(candidates)
        candidates_dict["rank"].extend(ranks)

    df_candidates_covisit = pl.DataFrame(candidates_dict)

    df_candidates_covisit_all = (
        df_candidates_covisit
        .filter(pl.col("type") == "orders")
        .drop("type")
        .explode(["candidates", "rank"])
        .rename({"candidates": "aid", "rank": "rank_orders"})
        .join(
            (
                df_candidates_covisit
                .filter(pl.col("type") == "carts")
                .drop("type")
                .explode(["candidates", "rank"])
                .rename({"candidates": "aid", "rank": "rank_carts"})
            ),
            on=["session", "aid"],
            how="outer"
        )
        .join(
            (
                df_candidates_covisit
                .filter(pl.col("type") == "clicks")
                .drop("type")
                .explode(["candidates", "rank"])
                .rename({"candidates": "aid", "rank": "rank_clicks"})
            ),
            on=["session", "aid"],
            how="outer"
        )
        .fill_null(999)
        .unique(subset=["session", "aid"], keep="last")
    )

    filename = (save_path / f"__{_dataset_type}__candidates_covisit_all_topk={topk}.parquet").as_posix()
    print("Saving to file...: ", filename)
    df_candidates_covisit_all.write_parquet(filename)


def generate_candidates_implicit(
    _dataset_type: str,
    _preprocess_type: str,
    _model_type: str,
    save_path: Path = Path("reranker_finale"),
    k: int = 100,
    topk: int = 100,
):
    df_train, df_test, df_target = None, None, None
    df = None
    print("Reading dataset...: ", _dataset_type)
    if _dataset_type == "valid":
        df_train = pl.read_parquet(VALIDATION_PATH / "train.parquet", use_pyarrow=True)
        df_test = pl.read_parquet(VALIDATION_PATH / "valid.parquet", use_pyarrow=True)
        df_target = pl.read_parquet(VALIDATION_PATH / "test_labels.parquet", use_pyarrow=True)
        pass
    elif _dataset_type == "subm":
        # read test data
        df_train = pl.read_parquet(TRAIN_PROCESSED, use_pyarrow=True)
        df_test = pl.read_parquet(TEST_PROCESSED, use_pyarrow=True)
    elif _dataset_type == "subm_no_fw":
        # read test data - cutted first week
        pass
        df_train = pl.read_parquet(TRAIN_PROCESSED, use_pyarrow=True)
        df_train = df_train.filter(pl.col("ts") >= datetime(2022, 8, 8).timestamp() * 1000)
        df_test = pl.read_parquet(TEST_PROCESSED, use_pyarrow=True)
    else:
        raise ValueError(f"Wrong _dataset_type {_dataset_type}")
    
    df = pl.concat([df_train, df_test]).unique().sort(["session", "ts"])
    print("Processing data...: ", _preprocess_type)
    if _preprocess_type == "old":
        df = implicit_old_weight_interactions(df)
    elif _preprocess_type == "new":
        df = implicit_new_weight_interactions(df)
    else:
        raise ValueError(f"Wrong _preprocess_type {_preprocess_type}")
    
    train_data = make_sparse_matrix(df)
    print("Training model...: ", _model_type)
    if _model_type == "i2i":
        model = CosineRecommender(K=k)
    elif _model_type == "tfidf":
        model = TFIDFRecommender(K=k)
    model.fit(train_data)

    test_users = df_test["session"].unique().to_list()
    print("Generating candidates...: ")
    df_candidates_i2i = implicit_batch_candidates_for_all_types(
        model=model, model_name=f"{_model_type}_{_preprocess_type}",
        train_data=train_data, test_users=test_users,
        topk=topk,
    )

    df_candidates_i2i = (
        df_candidates_i2i
        .explode(["aid", f"{_model_type}_{_preprocess_type}_score"])
        .filter(pl.col("aid") != -1)  # some strange items from implicit
        .unique(subset=["session", "aid"], keep="last")
    )

    filename = (save_path / f"__{_dataset_type}__candidates_{_model_type}_{_preprocess_type}_k={k}_topk={topk}.parquet").as_posix()
    print("Saving to file...: ", filename)
    df_candidates_i2i.write_parquet(filename)


def generate_dataset_from_candidates(
    candidate_files: list,
    mode: str = None,
    return_users_with_positives = False,
    action_type: str = None,
    version: str = None,
    save_path: Path = Path("reranker_finale"),
):
    df_target = None
    print("generating mode: ", mode)
    print("only return users with positives (for train mode only): ", return_users_with_positives)
    print("action type: ", action_type)
    if mode == "train":
        print("Reading target file")
        val_df_valid_input = pl.read_parquet(VALIDATION_PATH / "valid.parquet", use_pyarrow=True)
        df_target = (
            pl.read_parquet(VALIDATION_PATH / "test_labels.parquet", use_pyarrow=True)
            .filter(pl.col("type") == action_type)
            .join(val_df_valid_input.select(["session"]).unique(), on="session", how="inner")
            .drop("type")
            .explode("ground_truth")
            .with_column(pl.lit(1).alias("target"))
            .rename({"ground_truth": "aid"})
        )

    print("Reading candidate files...: ")
    df_candidates = None
    for cand_fname in candidate_files:
        filename = (save_path / cand_fname).as_posix()
        print("Current file: ", filename)
        df = pl.read_parquet(filename, columns=["session", "aid"])
        if df_candidates is None:
            df_candidates = df
        else:
            df_candidates = df_candidates.join(df, on=["session", "aid"], how="outer")
    
    print("Drop duplicate candidates, keeping last...: ")
    df_candidates = (
        df_candidates
        .filter(pl.col("aid") != -1)  # some strange items from implicit
        .unique(subset=["session", "aid"], keep="last")
    )

    if mode == "train":
        print("Adding target")
        df_candidates = (
            df_candidates
            .join(df_target, on=["session", "aid"], how="left")  # if using rank column as a feature
            .sort("session")
            .fill_null(0)
        )
        df_candidates = get_candidates_with_positives(df_candidates, df_target)

    filename = (save_path / f"__stage2__{mode}__{action_type}__candidates_{version}.parquet").as_posix()
    print("Saving to file...: ", filename)
    df_candidates.write_parquet(filename)

    # return df_candidates


def get_candidates_with_positives(df_candidates, df_target):
    df_sessions_with_positives = (
        df_candidates
        .groupby(["session"]).agg(pl.sum("target"))
        .filter(pl.col("target") > 0)
        .select(["session"])
    )

    print("sessions in test dataframe: ", df_candidates["session"].n_unique())
    print("sessions with positives in candidates: ", df_sessions_with_positives["session"].n_unique())
    print("sessions with positives in test dataframe: ", df_target["session"].n_unique())
    print(
        "hit rate: ", 
        df_sessions_with_positives["session"].n_unique() / df_target["session"].n_unique()
    )
    
    df_candidates_with_positives = df_candidates.join(df_sessions_with_positives, on="session", how="inner")
    print(df_candidates_with_positives["target"].value_counts())
    
    return df_candidates_with_positives
