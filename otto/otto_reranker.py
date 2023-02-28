import polars as pl
from tqdm import tqdm
from notebooks.otto.otto_word2vec import Word2Vec, w2v_cosine_sim


def cand_w2v_features(df_candidates, df_user_last_type_actions, w2vec):
    
    df_w2v_features = (
        df_candidates
        .join(df_user_last_type_actions, on="session", how="left")
    )

    for t in tqdm(["last_click_aid", "last_cart_aid", "last_order_aid"]):
        df_w2v_score = (
            df_w2v_features
            .filter((pl.col(t).is_not_null()))
            .select(["session", "aid", t])
        )

        l_aids = df_w2v_score["aid"].to_numpy()
        r_aids = df_w2v_score[t].to_numpy()

        cosine_sim_score = w2v_cosine_sim(w2vec, l_aids, r_aids)
        df_w2v_score = df_w2v_score.with_column(pl.Series(cosine_sim_score).alias(f"w2v_cosine_sim_{t}"))

        df_w2v_features = (
            df_w2v_features.join(df_w2v_score, on=["session", "aid", t], how="left")
            .with_column(pl.col(f"w2v_cosine_sim_{t}").fill_null(pl.lit(-999)))
        )
        
    df_w2v_features = df_w2v_features.select([
        "session", "aid", 
        "w2v_cosine_sim_last_click_aid", "w2v_cosine_sim_last_cart_aid", "w2v_cosine_sim_last_order_aid"
    ])
    
    return df_w2v_features


def cand_item_to_item_features(
        df_candidates, 
        df_user_last_type_actions,
        df_carts_orders,
        df_buys2buys,
        df_clicks,
    ):
    df_item_to_item_features = (
        df_candidates
        .join(df_user_last_type_actions, on="session", how="left")
        .join(
            df_carts_orders.rename({"weight": "user_last_click_aid_carts_orders_weight"}),
            left_on=["aid", "last_click_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_last_click_aid_buy2buy_weight"}),
            left_on=["aid", "last_click_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_last_click_aid_click_weight"}),
            left_on=["aid", "last_click_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_carts_orders.rename({"weight": "user_last_cart_aid_carts_orders_weight"}),
            left_on=["aid", "last_cart_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_last_cart_aid_buy2buy_weight"}),
            left_on=["aid", "last_cart_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_last_cart_aid_click_weight"}),
            left_on=["aid", "last_cart_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_carts_orders.rename({"weight": "user_last_order_aid_carts_orders_weight"}),
            left_on=["aid", "last_order_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_last_order_aid_buy2buy_weight"}),
            left_on=["aid", "last_order_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_last_order_aid_click_weight"}),
            left_on=["aid", "last_order_aid"], right_on=["aid", "aid_right"], how="left"
        )
        .drop(["last_click_aid", "last_cart_aid", "last_order_aid"])
        .fill_null(0)
    )
    return df_item_to_item_features


def cand_other_features(
    df_candidates: pl.DataFrame,
    df_user_action_stats_features,
    df_item_action_stats_features,
    df_item_n_sess_multiple_action,
    df_user_item_history_features,
):
    return (
        df_candidates
        .join(df_user_action_stats_features, on="session", how="left")
    #     .join(df_user_time_distr_features, on="session", how="left")
        .join(df_item_action_stats_features, on="aid", how="left")
    #     .join(df_item_time_distr_features, on="aid", how="left") # лишний признак
        .join(df_item_n_sess_multiple_action, on="aid", how="left")
        .join(df_user_item_history_features, on=["session", "aid"], how="left")
        .sort("session")
        .fill_null(0)
    )

