from datetime import datetime
import gc
import pandas as pd
import polars as pl
from tqdm import tqdm

from notebooks.otto.otto_word2vec import w2v_cosine_sim


def item_n_sessions_with_repeated_actions(df: pl.DataFrame):
    return (
        df
        .groupby(["session", "aid", "type"])
        .agg([pl.n_unique("ts").alias("n_actions")])
        .pivot(values="n_actions", index=["session", "aid"], columns="type")
        .groupby("aid")
        .agg([
            (
                pl.when(pl.col("0") > 1).then(pl.col("session")).otherwise(None).n_unique() - 1
            ).alias("item_n_sess_multi_clicks"),
            (
                pl.when(pl.col("1") > 1).then(pl.col("session")).otherwise(None).n_unique() - 1
            ).alias("item_n_sess_multi_carts"),
            (
                pl.when(pl.col("2") > 1).then(pl.col("session")).otherwise(None).n_unique() - 1
            ).alias("item_n_sess_multi_buys"),
        ])
    )


def user_time_distr_features(df: pl.DataFrame):
    df = df.join(df.groupby('session').agg([pl.count('aid').alias('sess_cnt')]), on='session', how='left')
    print(datetime.fromtimestamp(df['ts'].min() / 1000))
    print(datetime.fromtimestamp(df['ts'].max() / 1000))
    
    dts = pd.to_datetime((df['ts'] / 1000).to_pandas(), unit='s')
    df = (
        df
        .with_column(pl.from_pandas(dts.dt.weekday.rename('day')))
        .with_column(pl.from_pandas(dts.dt.hour.rename('hour')))
        .with_column(pl.from_pandas((dts.dt.hour*100 + dts.dt.minute*100//60).rename('hm')))
    )
    
    hm_stats = df.groupby('session').agg([
        pl.mean('hm').alias('user_hm_mean'),
        pl.median('hm').alias('user_hm_median'),
        pl.std('hm').fill_null(0).alias("user_hm_std")
    ])
    df = df.join(hm_stats, on='session', how='left')
    
    # days distr
    for i in range(7):
        df_temp_cnt = df.filter(pl.col('day') == i).groupby('session').count()
        df_temp_cnt.columns = ['session', f'user_day{i}cnt']
        df = df.join(df_temp_cnt, on='session', how='left')
        df = df.with_column(pl.col(f'user_day{i}cnt').fill_null(0))
        df = df.with_column(df[f'user_day{i}cnt']/df['sess_cnt'])
        del df_temp_cnt
        gc.collect()
    
    # hours distr
    for i in range(24):
        df_temp_cnt = df.filter(pl.col('hour') == i).groupby('session').count()
        df_temp_cnt.columns = ['session', f'user_hour{i}cnt']
        df = df.join(df_temp_cnt, on='session', how='left')
        df = df.with_column(pl.col(f'user_hour{i}cnt').fill_null(0))
        df = df.with_column(df[f'user_hour{i}cnt']/df['sess_cnt'])
        del df_temp_cnt
        gc.collect()
    
    df = (
        df.groupby('session').first().sort('session')
        .drop(['aid', 'sess_cnt', 'ts', 'type', 'day', 'hour', 'hm'])
    )
    
    return df


def item_time_distr_features(df: pl.DataFrame):
    df = df.join(df.groupby('aid').agg([pl.count('session').alias('aid_cnt')]), on='aid', how='left')
    print(datetime.fromtimestamp(df['ts'].min() / 1000))
    print(datetime.fromtimestamp(df['ts'].max() / 1000))
    
    dts = pd.to_datetime((df['ts'] / 1000).to_pandas(), unit='s')
    df = (
        df
        .with_column(pl.from_pandas(dts.dt.weekday.rename('day')))
        .with_column(pl.from_pandas(dts.dt.hour.rename('hour')))
        .with_column(pl.from_pandas((dts.dt.hour*100 + dts.dt.minute*100//60).rename('hm')))
    )
    
    hm_stats = df.groupby('aid').agg([
        pl.mean('hm').alias('item_hm_mean'),
        pl.median('hm').alias('item_hm_median'),
        pl.std('hm').fill_null(0).alias("item_hm_std")
    ])
    df = df.join(hm_stats, on='aid', how='left')
    
    # days distr
    for i in range(7):
        df_temp_cnt = df.filter(pl.col('day') == i).groupby('aid').count()
        df_temp_cnt.columns = ['aid', f'item_day{i}cnt']
        df = df.join(df_temp_cnt, on='aid', how='left')
        df = df.with_column(pl.col(f'item_day{i}cnt').fill_null(0))
        df = df.with_column(df[f'item_day{i}cnt']/df['aid_cnt'])
        del df_temp_cnt
        gc.collect()
    
    # hours distr
    for i in range(24):
        df_temp_cnt = df.filter(pl.col('hour') == i).groupby('aid').count()
        df_temp_cnt.columns = ['aid', f'item_hour{i}cnt']
        df = df.join(df_temp_cnt, on='aid', how='left')
        df = df.with_column(pl.col(f'item_hour{i}cnt').fill_null(0))
        df = df.with_column(df[f'item_hour{i}cnt']/df['aid_cnt'])
        del df_temp_cnt
        gc.collect()
    
    df = (
        df.groupby('aid').first().sort('aid')
        .drop(['session', 'aid_cnt', 'ts', 'type', 'day', 'hour', 'hm'])
    )
    
    return df


def user_action_stats_features(df: pl.DataFrame):
    
    pldf_user_type_stats = (
        df
        .groupby(["session", "type"])
        .agg([
            pl.count("aid").alias("cnt"),
            pl.n_unique("aid").alias("uniq_aids"),
        ])
    )

    pldf_user_type_uniq_aids = (
        pldf_user_type_stats
        .pivot(values="uniq_aids", index="session", columns="type")
        .rename({
            "0": "user_uniq_clicks",
            "1": "user_uniq_carts",
            "2": "user_uniq_orders",
        })
        .fill_null(0)
    )

    pldf_user_type_ratio = (
        pldf_user_type_stats
        .pivot(values="cnt", index="session", columns="type")
        .rename({
            "0": "cl_cnt",
            "1": "ca_cnt",
            "2": "or_cnt",
        })
        .with_columns([
            (pl.col('ca_cnt')/pl.col('cl_cnt')).alias('user_ca_cl_ratio'),
            (pl.col('or_cnt')/pl.col('cl_cnt')).alias('user_or_cl_ratio'),
            (pl.col('or_cnt')/pl.col('ca_cnt')).alias('user_or_ca_ratio'),
        ])
        .fill_null(0)
    )
    
    pldf_user_stats = (
        df
        .groupby("session")
        .agg([
            ( (pl.max("ts") - pl.min("ts")) / (1000 * 24 * 60 * 60)).alias("user_lifetime_days"),
            pl.count("aid").alias("user_n_actions"),
            pl.n_unique("aid").alias("user_n_uniq_items"),
            pl.mean("type").alias("user_buys_rate"),
        ])
        .join(pldf_user_type_uniq_aids, on="session", how="inner")
        .join(pldf_user_type_ratio, on="session", how="inner")
    )

    return pldf_user_stats


def item_action_stats_features(df: pl.DataFrame):
    
    pldf_item_type_stats = (
        df
        .groupby(["aid", "type"])
        .agg([
            pl.count("session").alias("cnt"),
            pl.n_unique("session").alias("uniq_sessions"),
        ])
    )

    pldf_item_type_uniq_sessions = (
        pldf_item_type_stats
        .pivot(values="uniq_sessions", index="aid", columns="type")
        .rename({
            "0": "item_uniq_clicks",
            "1": "item_uniq_carts",
            "2": "item_uniq_orders",
        })
        .fill_null(0)
    )

    pldf_item_type_ratio = (
        pldf_item_type_stats
        .pivot(values="cnt", index="aid", columns="type")
        .rename({
            "0": "cl_cnt",
            "1": "ca_cnt",
            "2": "or_cnt",
        })
        .with_columns([
            (pl.col('ca_cnt')/pl.col('cl_cnt')).alias('item_ca_cl_ratio'),
            (pl.col('or_cnt')/pl.col('cl_cnt')).alias('item_or_cl_ratio'),
            (pl.col('or_cnt')/pl.col('ca_cnt')).alias('item_or_ca_ratio'),
        ])
        .fill_null(0)
    )
    
    pldf_item_stats = (
        df
        .groupby("aid")
        .agg([
            ( (pl.max("ts") - pl.min("ts")) / (1000 * 24 * 60 * 60)).alias("item_lifetime_days"),
            pl.count("aid").alias("item_n_actions"),
            pl.n_unique("aid").alias("item_n_uniq_users"),
            pl.mean("type").alias("item_buys_rate"),
        ])
        .join(pldf_item_type_uniq_sessions, on="aid", how="inner")
        .join(pldf_item_type_ratio, on="aid", how="inner")
    )

    return pldf_item_stats


def user_item_history_features(df: pl.DataFrame):

    def add_action_num_reverse_chrono(df):
        return df.select([
            pl.col('*'),
            pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono')
        ])

    def add_session_length(df):
        return df.select([
            pl.col('*'),
            pl.col('session').count().over('session').alias('session_length')
        ])

    def add_log_recency_score(df):
        linear_interpolation = 0.1 + ((1-0.1) / (df['session_length']-1)) * (df['session_length']-df['action_num_reverse_chrono']-1)
        return df.with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)

    def add_type_weighted_log_recency_score(df):
        type_weights = {0:1, 1:6, 2:3}
        type_weighted_log_recency_score = pl.Series(df['log_recency_score'] / df['type'].apply(lambda x: type_weights[x]))
        return df.with_column(type_weighted_log_recency_score.alias('type_weighted_log_recency_score'))

    def apply(df, pipeline):
        for f in pipeline:
            df = f(df)
        return df
    
    pipeline = [
        add_action_num_reverse_chrono,
        add_session_length,
        add_log_recency_score,
        add_type_weighted_log_recency_score
    ]
    df_user_item_history_features = apply(df, pipeline)
    
    df_user_item_history_features = (
        df_user_item_history_features
        .groupby(["session", "aid"])
        .agg([
            pl.sum("log_recency_score").alias("user_item_log_recency_score"),
            pl.sum("type_weighted_log_recency_score").alias("user_item_type_weighted_log_recency_score"),
            pl.lit(1).alias("user_item_is_in_history")
        ])
        .sort("session")
    )
    
    return df_user_item_history_features


def cand_w2v_recent_item_features(df_candidates, df_user_recent_actions, w2vec):
    
    df_w2v_features = (
        df_candidates
        .join(df_user_recent_actions, on="session", how="left")
    )
    
    output_cols = []
    for t in tqdm(["t_0_recent", "t_-1_recent", "t_-2_recent", "t_-3_recent", "t_-4_recent"]):
        df_w2v_score = (
            df_w2v_features
            .filter((pl.col(t).is_not_null()))
            .select(["session", "aid", t])
        )

        l_aids = df_w2v_score["aid"].to_numpy()
        r_aids = df_w2v_score[t].to_numpy()

        cosine_sim_score = w2v_cosine_sim(w2vec, l_aids, r_aids)
        output_col = f"w2v_cosine_sim_{t}"
        output_cols.append(output_col)
        df_w2v_score = df_w2v_score.with_column(pl.Series(cosine_sim_score).alias(output_col))

        df_w2v_features = (
            df_w2v_features.join(df_w2v_score, on=["session", "aid", t], how="left")
            .with_column(pl.col(output_col).fill_null(pl.lit(-999)))
        )
        
    df_w2v_features = df_w2v_features.select(["session", "aid"] + output_cols)

    return df_w2v_features


def cand_recent_item_to_item_features(
        df_candidates, 
        df_user_recent_actions,
        df_carts_orders,
        df_buys2buys,
        df_clicks,
    ):
    df_item_to_item_features = (
        df_candidates
        .join(df_user_recent_actions, on="session", how="left")
        .join(
            df_carts_orders.rename({"weight": "user_t_0_recent_carts_orders_weight"}),
            left_on=["aid", "t_0_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_carts_orders.rename({"weight": "user_t_-1_recent_carts_orders_weight"}),
            left_on=["aid", "t_-1_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_carts_orders.rename({"weight": "user_t_-2_recent_carts_orders_weight"}),
            left_on=["aid", "t_-2_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_carts_orders.rename({"weight": "user_t_-3_recent_carts_orders_weight"}),
            left_on=["aid", "t_-3_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_carts_orders.rename({"weight": "user_t_-4_recent_carts_orders_weight"}),
            left_on=["aid", "t_-4_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_t_0_recent_buy2buy_weight"}),
            left_on=["aid", "t_0_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_t_-1_recent_buy2buy_weight"}),
            left_on=["aid", "t_-1_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_t_-2_recent_buy2buy_weight"}),
            left_on=["aid", "t_-2_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_t_-3_recent_buy2buy_weight"}),
            left_on=["aid", "t_-3_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_buys2buys.rename({"weight": "user_t_-4_recent_buy2buy_weight"}),
            left_on=["aid", "t_-4_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_t_0_recent_click_weight"}),
            left_on=["aid", "t_0_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_t_-1_recent_click_weight"}),
            left_on=["aid", "t_-1_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_t_-2_recent_click_weight"}),
            left_on=["aid", "t_-2_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_t_-3_recent_click_weight"}),
            left_on=["aid", "t_-3_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .join(
            df_clicks.rename({"weight": "user_t_-4_recent_click_weight"}),
            left_on=["aid", "t_-4_recent"], right_on=["aid", "aid_right"], how="left"
        )
        .drop(["t_0_recent", "t_-1_recent", "t_-2_recent", "t_-3_recent", "t_-4_recent"])
        .fill_null(0)
    )
    return df_item_to_item_features


def user_recent_actions(df):
    df_user_recent_actions = (
        df
        .groupby(["session"])
        .agg([
            pl.list("aid")
        ])
        .select([
            pl.col("session"),
            pl.col("aid").arr.get(-1).alias("t_0_recent"),
            pl.col("aid").arr.get(-2).alias("t_-1_recent"),
            pl.col("aid").arr.get(-3).alias("t_-2_recent"),
            pl.col("aid").arr.get(-4).alias("t_-3_recent"),
            pl.col("aid").arr.get(-5).alias("t_-4_recent"),
        ])
        .sort("session")
    )

    return df_user_recent_actions


def user_last_type_actions(df):
    df_user_last_type_actions = (
        df
        .groupby(["session", "type"])
        .agg([
            pl.last("aid")
        ])
        .pivot(values="aid", index="session", columns="type")
        .rename({
            "0": "last_click_aid",
            "1": "last_cart_aid",
            "2": "last_order_aid",
        })
        .sort("session")
    )

    return df_user_last_type_actions
