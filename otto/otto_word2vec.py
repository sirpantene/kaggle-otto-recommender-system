import hashlib
import numpy as np
import polars as pl
from gensim.models import Word2Vec


def hashf(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 32)


def get_w2v_embeddings(w2vec: Word2Vec):
    df_w2v_embeddings = pl.from_numpy(w2vec.wv.vectors).with_row_count()
    df_w2v_embeddings.columns = ["aid"] + [f"w2v_emb_{i}"
                                       for i in range(w2vec.wv.vectors.shape[1])]
    df_w2v_embeddings = df_w2v_embeddings.with_column(pl.col("aid").cast(pl.Int64))
    return df_w2v_embeddings


def w2v_cosine_sim(w2vec: Word2Vec, l_aids: list, r_aids: list):
    a_idx = [w2vec.wv.key_to_index[k] for k in l_aids]
    b_idx = [w2vec.wv.key_to_index[k] for k in r_aids]
    a_idx_embs = w2vec.wv.vectors[a_idx]
    b_idx_embs = w2vec.wv.vectors[b_idx]

    cosine_sim_score = (
        np.sum(a_idx_embs * b_idx_embs, axis=1) /
        (np.linalg.norm(a_idx_embs, axis=1) * np.linalg.norm(b_idx_embs, axis=1))
    )
    return cosine_sim_score
