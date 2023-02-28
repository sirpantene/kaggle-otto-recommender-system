import itertools
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from implicit.recommender_base import RecommenderBase
from tqdm import tqdm

def make_sparse_matrix(df: pl.DataFrame):
    row = df["session"].to_numpy()
    col = df["aid"].to_numpy()
    weight = df["weight"].to_numpy().astype(np.float32)
    
    return csr_matrix((weight, (row, col)))


from implicit.als import AlternatingLeastSquares

# als = AlternatingLeastSquares(
#     factors=100,
#     regularization=0.01,
#     alpha=1.0,
#     iterations=50,
#     calculate_training_loss=True,
#     use_gpu=True
# )

# als.fit(train_data)


def recommend():
    pass


def implicit_batch_submission(
    model: RecommenderBase, 
    train_data: csr_matrix, 
    test_users: list, 
    types: list = ("clicks", "carts", "orders"),  
    topk: int = 20, 
    b_sz: int = 1000
) -> pl.DataFrame:
    submission_dict = {"session_type": [], "labels": []}

    for test_session_start in tqdm(range(0, len(test_users), b_sz)):
        test_sessions = test_users[test_session_start : test_session_start + b_sz]
        rec_items, scores = model.recommend(
            test_sessions, user_items=train_data[test_sessions], N=topk,
            filter_already_liked_items=False, recalculate_user=False
        )
        session_types = [
            [f"{session_id}_{t}" for t in types]
            for session_id in test_sessions
        ]
        labels_list = [
            [" ".join(str(aid) for aid in recs.tolist())] * 3
            for recs in rec_items
        ]
        
        submission_dict["session_type"].extend(itertools.chain(*session_types))
        submission_dict["labels"].extend(itertools.chain(*labels_list))
        
    return pl.DataFrame(submission_dict)


def implicit_batch_candidates_for_all_types(
    model: RecommenderBase,
    model_name: str,
    train_data: csr_matrix, 
    test_users: list, 
    topk: int = 100, 
    b_sz: int = 1000
) -> pl.DataFrame:
    candidates_dict = {"session": [], "aid": [], f"{model_name}_score": []}

    for test_session_start in tqdm(range(0, len(test_users), b_sz)):
        test_sessions = test_users[test_session_start : test_session_start + b_sz]
        rec_items, scores = model.recommend(
            test_sessions, user_items=train_data[test_sessions], N=topk,
            filter_already_liked_items=False, recalculate_user=False
        )
    #     ranks = [
    #         np.arange(1, len(score) + 1).tolist()
    #         for score in scores
    #     ]

        candidates_dict["session"].extend(test_sessions)
        candidates_dict["aid"].extend(rec_items.tolist())
        candidates_dict[f"{model_name}_score"].extend(scores.tolist())
    #     submission_dict["i2i_rank"].extend(ranks)
        
    return pl.DataFrame(candidates_dict)


def implicit_old_weight_interactions(df: pl.DataFrame):
    df_action_weights = pl.DataFrame({"type": [0, 1, 2], "weight": [10, 30, 60]})

    return (
        df
        .join(df_action_weights, on="type", how="inner")
        .groupby(["session", "aid"])
        .agg([pl.sum("weight")])
    )

def implicit_new_weight_interactions(df: pl.DataFrame):
    df_type_weights = pl.DataFrame({"type": [0, 1, 2], "weight": [1, 3, 6]})

    df = (
        df
        .select([
            pl.col('*'),
            pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono'),
            pl.col('session').count().over('session').alias('session_length'),
        ])
    )
    linear_interpolation = 0.1 + ((1-0.1) / (df['session_length']-1)) * (df['session_length']-df['action_num_reverse_chrono']-1)
    return (
        df
        .with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)
        .join(df_type_weights, on="type")
        # .join(df_item_n_sess_multiple_action, on="aid", how="left")
        .with_column((pl.col("weight") * pl.col("log_recency_score")).alias("weight_recency"))
        .groupby(["session", "aid"])
        .agg([
            pl.sum("weight_recency").alias("weight")
        ])
    )
