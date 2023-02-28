from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from implicit.nearest_neighbours import CosineRecommender
from implicit.nearest_neighbours import TFIDFRecommender
from tqdm import tqdm
from notebooks.otto.otto_features import (
    cand_recent_item_to_item_features,
    cand_w2v_recent_item_features,
    item_action_stats_features,
    item_n_sessions_with_repeated_actions,
    item_time_distr_features,
    user_action_stats_features,
    user_item_history_features,
    user_last_type_actions,
    user_recent_actions, 
    user_time_distr_features
)
from notebooks.otto.otto_reranker import cand_item_to_item_features, cand_other_features, cand_w2v_features

from notebooks.otto.otto_utils import VALIDATION_PATH, TRAIN_PROCESSED, TEST_PROCESSED
from notebooks.otto.otto_candidates_covisit import CovisitationRecommender
from notebooks.otto.otto_implicit import (
    make_sparse_matrix,
    implicit_old_weight_interactions,
    implicit_new_weight_interactions,
    implicit_batch_candidates_for_all_types,
)
from notebooks.otto.otto_word2vec import hashf, Word2Vec


def generate_dataset_features():
    pass


def build_user_features(
    _dataset_type: str,
    save_path: Path = Path("reranker_finale"),
):
    df_test = None
    print("Reading dataset...: ", _dataset_type)
    if _dataset_type == "valid":
        df_test = pl.read_parquet(VALIDATION_PATH / "valid.parquet", use_pyarrow=True)
    elif _dataset_type == "subm":
        df_test = pl.read_parquet(TEST_PROCESSED, use_pyarrow=True)
    else:
        raise ValueError(f"Wrong _dataset_type {_dataset_type}")
    
    df = df_test.unique().sort(["session", "ts"])

    df_user_action_stats_features = user_action_stats_features(df)
    df_user_time_distr_features = user_time_distr_features(df)
    df_user_item_history_features = user_item_history_features(df)
    df_user_last_type_actions = user_last_type_actions(df)
    df_user_recent_actions = user_recent_actions(df)
    df_user_action_stats_features.write_parquet(save_path / f"__features__{_dataset_type}__user_action_stats.parquet")
    df_user_time_distr_features.write_parquet(save_path / f"__features__{_dataset_type}__user_time_distr.parquet")
    df_user_item_history_features.write_parquet(save_path / f"__features__{_dataset_type}__user_item_history.parquet")
    df_user_last_type_actions.write_parquet(save_path / f"__features__{_dataset_type}__user_last_type_actions.parquet")
    df_user_recent_actions.write_parquet(save_path / f"__features__{_dataset_type}__user_recent_actions.parquet")


def build_item_features(
    _dataset_type: str,
    save_path: Path = Path("reranker_finale"),
):
    df_train, df_test, df_target = None, None, None
    df = None
    print("Reading dataset...: ", _dataset_type)
    if _dataset_type == "valid":
        df_train = pl.read_parquet(VALIDATION_PATH / "train.parquet", use_pyarrow=True)
        df_test = pl.read_parquet(VALIDATION_PATH / "valid.parquet", use_pyarrow=True)
        pass
    elif _dataset_type == "subm":
        # read test data
        df_train = pl.read_parquet(TRAIN_PROCESSED, use_pyarrow=True)
        df_train = df_train.filter(pl.col("ts") >= datetime(2022, 8, 8).timestamp() * 1000)
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

    df_item_action_stats_features = item_action_stats_features(df)
    df_item_time_distr_features = item_time_distr_features(df)
    df_item_n_sess_multiple_action = item_n_sessions_with_repeated_actions(df)
    df_item_action_stats_features.write_parquet(save_path / f"__features__{_dataset_type}__item_action_stats.parquet")
    df_item_time_distr_features.write_parquet(save_path / f"__features__{_dataset_type}__item_time_distr.parquet")
    df_item_n_sess_multiple_action.write_parquet(save_path / f"__features__{_dataset_type}__item_n_sess_multiple_action.parquet")


def generate_dataset_feature_group_cand_user_item(
    mode: str = None,
    action_type: str = None,
    version: str = None,
    save_path: Path = Path("reranker_finale"), 
):
    print("mode: ", mode)
    print("action type: ", action_type)
    print("version: ", version)
    filename = (save_path / f"__stage2__{mode}__{action_type}__candidates_{version}.parquet").as_posix()
    print("Reading generated candidate file...: ", filename)
    df_candidates = pl.read_parquet(filename, columns=["session", "aid"])

    feature_group = "cand_user_item"
    _dataset_type = "valid" if mode == "train" else "subm"
    _settings = {
        "files": [
            f"__{_dataset_type}__candidates_covisit_all_topk=200.parquet",
            f"__{_dataset_type}__candidates_tfidf_new_k=200_topk=100.parquet",
            f"__{_dataset_type}__candidates_tfidf_old_k=200_topk=100.parquet",
            f"__{_dataset_type}__candidates_i2i_new_k=100_topk=100.parquet",
            f"__{_dataset_type}__candidates_i2i_old_k=100_topk=100.parquet",
        ],
        "key": ["session", "aid"],
        "save_file": f"__dataset__features__{mode}__{action_type}__{version}_features_{feature_group}.parquet",
    }

    df_features = []
    for fname in tqdm(_settings["files"]):
        print("Processing feature file...: ", fname)
        df = pl.read_parquet(save_path / fname)
        df = (
            df_candidates
            .join(df, on=_settings["key"], how="left")
            .fill_null(999)
        )
        df_features.append(df)

    def apply(df_root, df_append_list):
        df = df_root
        for df_a in df_append_list:
            df = df.join(df_a, on=["session", "aid"], how="left")
        return df
    
    df_features = apply(df_candidates, df_features)
    
    filename = (save_path / _settings["save_file"]).as_posix()
    print("Saving to file...: ", filename)
    df_features.write_parquet(filename)


def generate_dataset_feature_group_w2v_item2item(
    mode: str = None,
    action_type: str = None,
    version: str = None,
    save_path: Path = Path("reranker_finale"),    
):
    print("mode: ", mode)
    print("action type: ", action_type)
    print("version: ", version)
    filename = (save_path / f"__stage2__{mode}__{action_type}__candidates_{version}.parquet").as_posix()
    print("Reading generated candidate file...: ", filename)
    df_candidates = pl.read_parquet(filename, columns=["session", "aid"])

    feature_group = "w2v_item2item"
    _settings = {
        "save_file": f"__dataset__features__{mode}__{action_type}__{version}_features_{feature_group}.parquet",
    }
    _dataset_type = "valid" if mode == "train" else "subm"
    df_user_last_type_actions = pl.read_parquet(save_path / f"__features__{_dataset_type}__user_last_type_actions.parquet")
    df_user_recent_actions = pl.read_parquet(save_path / f"__features__{_dataset_type}__user_recent_actions.parquet")

    if _dataset_type == "valid":
        w2vec = Word2Vec.load((save_path / "__valid__word2vec_window=10_negative=20.w2v").as_posix())
    elif _dataset_type == "subm":
        w2vec = Word2Vec.load((save_path / "__subm__word2vec_window=10_negative=20.w2v").as_posix())
    elif _dataset_type == "subm_no_fw":
        pass
        # read test data - cutted first week
        # carts_orders = pl.read_parquet(save_path / "__valid__covisit_carts_orders_all_v3.parquet")
        # buys2buys = pl.read_parquet(save_path / "__valid__covisit_buys2buys_all_v4.parquet")
        # clicks = pl.read_parquet(save_path / "__valid__covisit_clicks_all_v3.parquet")
    else:
        raise ValueError(f"Wrong _dataset_type {_dataset_type}")

    print("Processing recent items features...")
    df_cand_w2v_recent_item_features = cand_w2v_recent_item_features(
        df_candidates,
        df_user_recent_actions,
        w2vec
    )

    print("Processing last type items features...")
    df_cand_w2v_features = cand_w2v_features(
        df_candidates,
        df_user_last_type_actions,
        w2vec
    )

    df_features = (
        df_cand_w2v_recent_item_features
        .join(df_cand_w2v_features, on=["session", "aid"], how="inner")
    )

    filename = (save_path / _settings["save_file"]).as_posix()
    print("Saving to file...: ", filename)
    df_features.write_parquet(filename)


def generate_dataset_feature_group_covisit_item2item(
    mode: str = None,
    action_type: str = None,
    version: str = None,
    save_path: Path = Path("reranker_finale"),    
):
    print("mode: ", mode)
    print("action type: ", action_type)
    print("version: ", version)
    filename = (save_path / f"__stage2__{mode}__{action_type}__candidates_{version}.parquet").as_posix()
    print("Reading generated candidate file...: ", filename)
    df_candidates = pl.read_parquet(filename, columns=["session", "aid"])

    feature_group = "covisit_item2item"
    _settings = {
        "save_file": f"__dataset__features__{mode}__{action_type}__{version}_features_{feature_group}.parquet",
    }
    _dataset_type = "valid" if mode == "train" else "subm"
    df_user_last_type_actions = pl.read_parquet(save_path / f"__features__{_dataset_type}__user_last_type_actions.parquet")
    df_user_recent_actions = pl.read_parquet(save_path / f"__features__{_dataset_type}__user_recent_actions.parquet")

    if _dataset_type == "valid":
        carts_orders = pl.read_parquet(save_path / "__valid__covisit_carts_orders_all_v3.parquet")
        buys2buys = pl.read_parquet(save_path / "__valid__covisit_buys2buys_all_v4.parquet")
        clicks = pl.read_parquet(save_path / "__valid__covisit_clicks_all_v3.parquet")
    elif _dataset_type == "subm":
        carts_orders = pl.read_parquet(save_path / "__subm__covisit_carts_orders_all_v1.parquet")
        buys2buys = pl.read_parquet(save_path / "__subm__covisit_buys2buys_all_v1.parquet")
        clicks = pl.read_parquet(save_path / "__subm__covisit_clicks_all_v1.parquet")
    elif _dataset_type == "subm_no_fw":
        pass
        # read test data - cutted first week
        # carts_orders = pl.read_parquet(save_path / "__valid__covisit_carts_orders_all_v3.parquet")
        # buys2buys = pl.read_parquet(save_path / "__valid__covisit_buys2buys_all_v4.parquet")
        # clicks = pl.read_parquet(save_path / "__valid__covisit_clicks_all_v3.parquet")
    else:
        raise ValueError(f"Wrong _dataset_type {_dataset_type}")

    print("Processing recent items features...")
    df_cand_recent_item_to_item_features = cand_recent_item_to_item_features(
        df_candidates,
        df_user_recent_actions,
        df_carts_orders=carts_orders,
        df_buys2buys=buys2buys,
        df_clicks=clicks
    )

    print("Processing last type items features...")
    df_item_to_item_features = cand_item_to_item_features(
        df_candidates,
        df_user_last_type_actions,
        df_carts_orders=carts_orders,
        df_buys2buys=buys2buys,
        df_clicks=clicks
    )

    df_features = (
        df_cand_recent_item_to_item_features
        .join(df_item_to_item_features, on=["session", "aid"], how="inner")
    )

    filename = (save_path / _settings["save_file"]).as_posix()
    print("Saving to file...: ", filename)
    df_features.write_parquet(filename)


def generate_dataset_feature_group_other(
    mode: str = None,
    action_type: str = None,
    version: str = None,
    save_path: Path = Path("reranker_finale"),
):
    print("mode: ", mode)
    print("action type: ", action_type)
    print("version: ", version)
    filename = (save_path / f"__stage2__{mode}__{action_type}__candidates_{version}.parquet").as_posix()
    print("Reading generated candidate file...: ", filename)
    df_candidates = pl.read_parquet(filename, columns=["session", "aid"])

    feature_group = "other"
    _settings = {
        "save_file": f"__dataset__features__{mode}__{action_type}__{version}_features_{feature_group}.parquet",
    }
    
    _dataset_type = "valid" if mode == "train" else "subm"
    df_user_action_stats_features = pl.read_parquet(save_path / f"__features__{_dataset_type}__user_action_stats.parquet")
    # df_user_time_distr_features = pl.read_parquet(save_path / f"__features__{_dataset_type}__user_time_distr.parquet")
    df_user_item_history_features = pl.read_parquet(save_path / f"__features__{_dataset_type}__user_item_history.parquet")
    df_item_action_stats_features = pl.read_parquet(save_path / f"__features__{_dataset_type}__item_action_stats.parquet")
    # df_item_time_distr_features = pl.read_parquet(save_path / f"__features__{_dataset_type}__item_time_distr.parquet")
    df_item_n_sess_multiple_action = pl.read_parquet(save_path / f"__features__{_dataset_type}__item_n_sess_multiple_action.parquet")

    df_other_features = cand_other_features(
        df_candidates.select(["session", "aid"]),
        df_user_action_stats_features,
        df_item_action_stats_features,
        df_item_n_sess_multiple_action,
        df_user_item_history_features,
    )

    filename = (save_path / _settings["save_file"]).as_posix()
    print("Saving to file...: ", filename)
    df_other_features.write_parquet(filename)


def gather_dataset_features(
    mode: str = None,
    action_type: str = None,
    version: str = None,
    feature_groups: list = None,
    save_path: Path = Path("reranker_finale"),
    batch_users: list = None,
):
    print("mode: ", mode)
    print("action type: ", action_type)
    print("version: ", version)
    filename = (save_path / f"__stage2__{mode}__{action_type}__candidates_{version}.parquet").as_posix()
    print("Reading generated candidate file...: ", filename)
    df_candidates = pl.read_parquet(filename)

    if batch_users is not None:
        df_candidates = df_candidates.filter(pl.col("session").is_in(batch_users))

    df_dataset = df_candidates
    for feature_group in feature_groups:
        fg_path = save_path / f"__dataset__features__{mode}__{action_type}__{version}_features_{feature_group}.parquet"
        df_feature_group = pl.read_parquet(fg_path)
        df_dataset = df_dataset.join(df_feature_group, on=["session", "aid"], how="left")
    df_dataset = df_dataset.fill_null(0)

    return df_dataset
