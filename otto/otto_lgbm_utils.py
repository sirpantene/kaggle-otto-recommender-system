from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
from numba import njit, prange
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from notebooks.otto.otto_utils import calc_valid_score


def recall20(preds, targets, groups):
    total = 0
    nonempty = 0
    group_starts = np.cumsum(groups)

    for group_id in range(len(groups)):
        group_end = group_starts[group_id]
        group_start = group_end - groups[group_id]
        ranks = np.argsort(preds[group_start:group_end])[::-1]
        hits = 0
        for i in range(min(len(ranks), 20)):
            hits += targets[group_start + ranks[i]]

        actual = min(20, targets[group_start:group_end].sum())
        if actual > 0:
            total += hits / actual
            nonempty += 1

    return total / nonempty


# custom metric for LightGBM should return 
# "metric name", "metric value" and "greater is better" flag
def lgb_recall(preds, lgb_dataset):
    metric = recall20(preds, lgb_dataset.label, lgb_dataset.group)
    return 'recall@20', metric, True


@njit() # the only difference from previous version
def numba_recall20(preds, targets, groups):
    total = 0
    nonempty = 0
    group_starts = np.cumsum(groups)

    for group_id in range(len(groups)):
        group_end = group_starts[group_id]
        group_start = group_end - groups[group_id]
        ranks = np.argsort(preds[group_start:group_end])[::-1]
        hits = 0
        for i in range(min(len(ranks), 20)):
            hits += targets[group_start + ranks[i]]

        actual = min(20, targets[group_start:group_end].sum())
        total += hits
        nonempty += actual
        # if actual > 0:
        #     total += hits / actual
        #     nonempty += 1
        

    return total / nonempty


def lgb_numba_recall(preds, lgb_dataset):
    metric = numba_recall20(preds, lgb_dataset.label, lgb_dataset.group)
    return 'numba_recall@20', metric, True


@njit(parallel=True) # added parallel flag
def numba_parallel_recall20(preds, targets, groups):
    total = 0
    nonempty = 0
    group_starts = np.cumsum(groups)

    for group_id in prange(len(groups)): # changed range to prange
        group_end = group_starts[group_id]
        group_start = group_end - groups[group_id]
        ranks = np.argsort(preds[group_start:group_end])[::-1]
        hits = 0
        for i in range(min(len(ranks), 20)):
            hits += targets[group_start + ranks[i]]

        actual = min(20, targets[group_start:group_end].sum())
        if actual > 0:
            total += hits / actual
            nonempty += 1

    return total / nonempty


def calc_oof_score_for_type(df_valid_preds: pl.DataFrame, act_type: str):
    scores = (
        df_valid_preds
        .select([
            pl.col("scores_fold0"),
            pl.col("scores_fold1"),
            pl.col("scores_fold2"), 
            pl.col("scores_fold3"), 
            pl.col("scores_fold4"),
        ]).mean(axis=1)
    )

    df_valid_preds_sorted = (
        df_valid_preds
        .with_column(scores.alias("score"))
        .select(["session", "aid", "score"])
        .sort("score", reverse=True)
        .groupby("session")
        .agg([
            pl.list("aid"),
            pl.list("score"),
        ])
    )

    df_submission_valid_type_reranked = lgb_oof_submission(df_valid_preds_sorted, act_type)
    return calc_valid_score(df_submission_valid_type_reranked, topk=20)


def lgb_oof_submission(df_oof_preds_sorted: pl.DataFrame, act_type: str):
    """
    df_oof_preds_sorted:
    ┌──────────┬───────────────────────────────┬─────────────────────────────────────┐
    │ session  ┆ aid                           ┆ score                               │
    │ ---      ┆ ---                           ┆ ---                                 │
    │ i64      ┆ list[i64]                     ┆ list[f64]                           │
    ╞══════════╪═══════════════════════════════╪═════════════════════════════════════╡
    │ 12309632 ┆ [1246180, 269352, ... 152547] ┆ [4.266326, 4.083894, ... -9.9461... │
    """
    submission_dict = {"session_type": [], "labels": []}
    types = [act_type]
    topk = 20

    for row in tqdm(df_oof_preds_sorted.rows()):
        session_id = row[0]
        rec_items = row[1][:topk]
        
        session_types = [f"{session_id}_{t}" for t in types]
        labels = " ".join(str(aid) for aid in rec_items)
        labels_list = [labels]
        
        submission_dict["session_type"].extend(session_types)
        submission_dict["labels"].extend(labels_list)

    return pl.DataFrame(submission_dict)


def lgb_train_cv(
    df_dataset: pl.DataFrame,
    action_type: str,
    candidates_version: str,
    negative_sampling: int = None, 
    model_version: str = "v0",
    save_path: Path = Path("reranker_finale"),
):

    default_params = {
        'boosting_type': 'gbdt',
    #     'objective': 'binary',
        'objective': 'lambdarank',
        'metric': '"None"',
        'eval_at': 20,
    #     'metric': {'auc', 'binary_logloss'},
    #     'min_data_in_leaf': 256, 
    #     'num_leaves': 63,
        'max_depth': 7,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'device': 'gpu',
        'verbose': -1,
        'num_threads': 44,
    }

    feature_cols = df_dataset.drop(["session", "aid", "target"]).columns
    print("num features: ", len(feature_cols))

    skf = GroupKFold(n_splits=5)
    df_oof_preds = df_dataset.select(["session", "aid"])

    if negative_sampling is not None and isinstance(negative_sampling, int):
        print("Applying negative sampling...")
        print("Dataset shape before negative sampling: ", df_dataset.shape)
        n_negatives = negative_sampling
        seed = 42
        df_session_n_positives = df_dataset.filter(pl.col("target") == 1).groupby("session").agg([pl.count("aid").alias("n_positives")])
        df_dataset_train = (
            df_dataset
            .join(df_session_n_positives, on="session")
            .with_column((pl.col("n_positives") * pl.lit(n_negatives)).alias("n_negatives"))
            .with_column(pl.arange(0, pl.count()).shuffle(seed=seed).over("session").alias("cand_id"))
            .filter(
                (pl.col("cand_id") <= pl.col("n_negatives")) |
                pl.col("target") == 1
            )
            .drop(["n_positives", "n_negatives", "cand_id"])
        )
        print("Dataset shape after negative sampling: ", df_dataset_train.shape)
    
    else:
        df_dataset_train = df_dataset

    for fold, (train_idx, valid_idx) in tqdm(enumerate(skf.split(df_dataset_train, df_dataset_train['target'], groups=df_dataset_train['session']))):
        # contains sessions with no positives
        X_train = df_dataset_train[train_idx][feature_cols].to_pandas()
        y_train = df_dataset_train[train_idx]["target"].to_pandas()
        X_valid = df_dataset_train[valid_idx][feature_cols].to_pandas()
        y_valid = df_dataset_train[valid_idx]["target"].to_pandas()
        
        # create dataset for lightgbm
        groups_len_train = (
            df_dataset_train[train_idx]
            .groupby("session").agg(pl.count("aid"))
            .sort("session")["aid"].to_numpy()
        )
        groups_len_valid = (
            df_dataset_train[valid_idx]
            .groupby("session").agg(pl.count("aid"))
            .sort("session")["aid"].to_numpy()
        )    
        lgb_train = lgb.Dataset(X_train, y_train, group=groups_len_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, group=groups_len_valid, reference=lgb_train)
        
        params = default_params
        gbm_ranking = lgb.train(
            params, 
            lgb_train,
            num_boost_round=20,
            feval=lgb_numba_recall,
            valid_sets=lgb_eval,
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=20)]
        )
        import gc
        del X_train, X_valid, lgb_train, lgb_eval
        gc.collect()

        X_pred = (
            df_dataset.join(
                df_dataset_train[train_idx].select(["session", "aid", pl.lit(1).alias("is_train")]),
                on=["session", "aid"],
                how="left"
            )
            .filter(pl.col("is_train") != 1)
        )
        print("predicting scores for out of fold items")
        # make batch preds:
        scores = gbm_ranking.predict(X_pred[feature_cols].to_numpy())
        # scores = gbm_ranking.predict(X_valid)
        df_valid_preds_fold = (
            X_pred.select(["session", "aid"])
            .with_columns([pl.Series(scores).alias(f"scores_fold{fold}")])
        )
        del X_pred
        df_oof_preds = df_oof_preds.join(df_valid_preds_fold, on=["session", "aid"], how="left")

        model_fname = f"__model__{action_type}_{candidates_version}_reranker_fold{fold}_{model_version}.lgb"
        gbm_ranking.save_model((save_path / model_fname).as_posix())
    
    return df_oof_preds


def lgb_inference(
    df,
    b_sz: int = 100_000,
):
    pass


def lgb_cv_folds_predictions(
    df,
    model_file_tmplt,
    action_type,
    save_path: Path = Path("reranker_finale"),
):
    feature_cols = df.drop(["session", "aid"]).columns
    df_valid_preds = df.select(["session", "aid"])
    X_test = df[feature_cols].to_numpy()

    for fold in tqdm(range(5)):
#         model_file = model_file_tmplt.format(action_type=action_type, fold=fold)
        model_file = model_file_tmplt.format(fold=fold)
        gbm_ranking = lgb.Booster(model_file=(save_path / model_file).as_posix())
        scores = gbm_ranking.predict(X_test)
        df_valid_preds_fold = (
            df.select(["session", "aid"])
            .with_columns([pl.Series(scores).alias(f"scores_fold{fold}")])
        )
        df_valid_preds = (
            df_valid_preds.join(df_valid_preds_fold, on=["session", "aid"], how="left")
        )
    
    scores = (
        df_valid_preds
        .select([
            pl.col("scores_fold0"),
            pl.col("scores_fold1"),
            pl.col("scores_fold2"), 
            pl.col("scores_fold3"), 
            pl.col("scores_fold4"),
        ]).mean(axis=1)
    )
    
    df_valid_preds = (
        df_valid_preds.with_column(scores.alias(f"{action_type}_score"))
        .select(["session", "aid", f"{action_type}_score"])
    )
    return df_valid_preds
