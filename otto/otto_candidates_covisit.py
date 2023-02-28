import itertools
from collections import Counter

import numpy as np
import polars as pl
from tqdm import tqdm


def weights_func_carts_order(df: pl.DataFrame, _type_weight=None):
    if _type_weight is None:
        _type_weight = {"type_right": [0, 1, 2], "weight": [0.5, 9, 0.5]}
    df_type_weight = pl.DataFrame(_type_weight)
    df = df.join(df_type_weight, on="type_right")
    return df


def weights_func_orders(df):
    df = df.with_column(pl.lit(1).alias("weight"))
    return df


def weights_func_clicks(df, _type="valid"):
    if _type == "valid":
        df = df.with_column(
            (1 + 3 * (pl.col("ts") - 1659304800025) / (1661723998621 - 1659304800025)).alias("weight")
        )
    elif _type == "test":
        df = df.with_column(
            (1 + 3 * (pl.col("ts") - 1659304800025) / (1662328791563 - 1659304800025)).alias("weight")
        )
    # 1659304800025 : minimum timestamp
    # 1661723998621 : maximum timestamp valid
    # 1662328791563 : maximum timestamp test
    return df


def drop_dupli_cols_for_carts_orders(df):
    df = df.select(['session', 'aid', 'aid_right', 'type_right']).unique()
    return df


def drop_dupli_cols_for_clicks(df):
    df = (
        df
        .groupby(['session', 'aid', 'aid_right'])
        .agg([pl.first('ts').alias("ts")])
    )
    return df


def co_visitation_matrix(
    df: pl.DataFrame,
    weights_func,
    drop_dupli_func,
    weights_func_params=None,
    chunk_size=200_000,
    drop_th_sess_num=30,
    attention_period=24*60*60*1000,
    attention_type=None,
    save_top_k=20,
):
    for session in tqdm(range(0, df["session"].max(), chunk_size)):
        s_start = session
        s_end = session + chunk_size - 1

        df_chunk = df.filter(pl.col("session").is_between(s_start, s_end))
        if attention_type is not None:
            df_chunk = df_chunk.filter(pl.col("type").is_in(attention_type))
        df_chunk = (
            df_chunk
            .sort(["session", "ts"], reverse=[False, True])
            .with_column(pl.lit(1).alias("ones"))
            .with_column(pl.col("ones").cumsum().over("session").alias("rank"))
            .select([pl.all().exclude("ones"),])
            .filter(pl.col("rank") <= drop_th_sess_num)
        )
        
        # create pairs
        df_chunk = df_chunk.join(df_chunk, on="session", how="inner")
        df_chunk = (
            df_chunk
            .filter(( (pl.col("ts") - pl.col("ts_right")) < attention_period) & 
                    (pl.col("aid") != pl.col("aid_right")))
        )
        
        df_chunk = drop_dupli_func(df_chunk)

        if weights_func_params is not None:
            df_chunk = weights_func(df_chunk, **weights_func_params)
        else:
            df_chunk = weights_func(df_chunk)

        df_chunk = (
            df_chunk
            .groupby(['aid', 'aid_right'])
            .agg([pl.sum("weight")])
        )

        if session == 0: tmp = df_chunk
        else: tmp = pl.concat([tmp, df_chunk]).groupby(['aid', 'aid_right']).agg([pl.sum("weight")])

    tmp = tmp.sort(['aid','weight'], reverse=[False, True])
    
    tmp_top_k = (
        tmp
        .with_column(pl.lit(1).alias("ones"))
        .with_column(pl.col("ones").cumsum().over("aid").alias("rank"))
        .filter(pl.col("rank") <= save_top_k)
    )
    return tmp, tmp_top_k


class CovisitationRecommender:

    def __init__(
        self,
        df_top_k_buys,
        df_top_k_buy2buy,
        df_top_k_clicks,
        top_carts,
        top_orders,
        top_clicks,
    ) -> None:
        self.type_weight_multipliers = {0: 0.5, 1: 9, 2: 0.5}
        self.top_k_buys = self.pldf_to_dict(df_top_k_buys)  # carts covisitation stats
        self.top_k_buy2buy = self.pldf_to_dict(df_top_k_buy2buy)  # orders covisitation stats
        self.top_k_clicks = self.pldf_to_dict(df_top_k_clicks)  # clicks covisitation stats
        self.top_carts = top_carts
        self.top_orders = top_orders
        self.top_clicks = top_clicks

    @staticmethod
    def pldf_to_dict(df: pl.DataFrame):
        df = df.groupby("aid").agg([pl.list("aid_right")])
        return dict(zip(df["aid"].to_list(), df["aid_right"].to_list()))

    def recommend_clicks(self, session_aid_list, session_type_list, topk=20):
        aids = session_aid_list
        types = session_type_list
        unique_aids = list(dict.fromkeys(aids[::-1]))

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= topk:
            weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
            aids_temp = Counter() 
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid,w,t in zip(aids, weights, types): 
                aids_temp[aid] += w * self.type_weight_multipliers[t]
            sorted_aids = [k for k, v in aids_temp.most_common(topk)]
            return sorted_aids

        # USE "CLICKS" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_k_clicks[aid] for aid in unique_aids if aid in self.top_k_clicks]))
        # RERANK CANDIDATES
        top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(topk) if aid2 not in unique_aids]    
        result = unique_aids + top_aids2[:topk - len(unique_aids)]
        set_result = set(result)  # remove duplicates
        return result + [i for i in self.top_clicks if i not in set_result][:topk - len(result)]
        
        # USE TOP20 TEST CLICKS
    #     return result + list(top_clicks)[:topk-len(result)]

    def recommend_carts(self, session_aid_list, session_type_list, topk=20):
        aids = session_aid_list
        types = session_type_list
        unique_aids = list(dict.fromkeys(aids[::-1]))

        buy_aids = [aid for i, aid in enumerate(aids) if types[i] != 2]
        buy_types = [t for i, t in enumerate(types) if types[i] != 2]
        unique_buys = list(dict.fromkeys(buy_aids[::-1]))

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= topk:
            weights=np.logspace(0.5,1,len(aids),base=2, endpoint=True)-1
            aids_temp = Counter() 
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid,w,t in zip(aids, weights, types): 
                aids_temp[aid] += w * self.type_weight_multipliers[t]

            # Rerank candidates using"top_20_carts" co-visitation matrix
            aids2 = list(itertools.chain(*[self.top_k_buys[aid] for aid in unique_buys if aid in self.top_k_buys]))
            for aid in aids2: aids_temp[aid] += 0.1
            sorted_aids = [k for k, v in aids_temp.most_common(topk)]
            return sorted_aids

        # Use "cart order" and "clicks" co-visitation matrices
        aids1 = list(itertools.chain(*[self.top_k_clicks[aid] for aid in unique_aids if aid in self.top_k_clicks]))
        aids2 = list(itertools.chain(*[self.top_k_buys[aid] for aid in unique_aids if aid in self.top_k_buys]))

        # RERANK CANDIDATES
        top_aids2 = [aid2 for aid2, cnt in Counter(aids1+aids2).most_common(topk) if aid2 not in unique_aids]    
        result = unique_aids + top_aids2[:topk - len(unique_aids)]
        set_result = set(result)  # remove duplicates
        return result + [i for i in self.top_carts if i not in set_result][:topk - len(result)]
        # USE TOP20 TEST CLICKS
    #     return result + list(top_clicks)[:topk-len(result)]

    def recommend_buys(self, session_aid_list, session_type_list, topk=20):
        aids = session_aid_list
        types = session_type_list
        unique_aids = list(dict.fromkeys(aids[::-1]))
        
        buy_aids = [aid for i, aid in enumerate(aids) if types[i] != 0]
        buy_types = [t for i, t in enumerate(types) if types[i] != 0]
        unique_buys = list(dict.fromkeys(buy_aids[::-1]))

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids)>=topk:
            weights=np.logspace(0.5,1,len(aids),base=2, endpoint=True)-1
            aids_temp = Counter() 
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid,w,t in zip(aids,weights,types): 
                aids_temp[aid] += w * self.type_weight_multipliers[t]
            # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
            aids3 = list(itertools.chain(*[self.top_k_buy2buy[aid] for aid in unique_buys if aid in self.top_k_buy2buy]))
            for aid in aids3: aids_temp[aid] += 0.1
            sorted_aids = [k for k,v in aids_temp.most_common(topk)]
            return sorted_aids

        # USE "CART ORDER" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_k_buys[aid] for aid in unique_aids if aid in self.top_k_buys]))
        # USE "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_k_buy2buy[aid] for aid in unique_buys if aid in self.top_k_buy2buy]))
        # RERANK CANDIDATES
        top_aids2 = [aid2 for aid2, cnt in Counter(aids2+aids3).most_common(topk) if aid2 not in unique_aids] 
        result = unique_aids + top_aids2[:topk - len(unique_aids)]
        set_result = set(result)  # remove duplicates
        return result + [i for i in self.top_orders if i not in set_result][:topk - len(result)]

        # USE TOP20 TEST ORDERS
    #     return result + list(top_orders)[:topk-len(result)]