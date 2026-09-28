
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, SVDpp, KNNBaseline, AlgoBase
from surprise.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import nnls
import lightgbm as lgb
from scipy.sparse.linalg import svds

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def load_data(dir_: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(dir_ / "train_data_movie_rate.csv")
    test = pd.read_csv(dir_ / "test_data.csv")
    return ratings, test


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["user_id", "item_id", "label"]).copy()
    df["label"] = df["label"].astype(float)
    return df.groupby(["user_id", "item_id"], as_index=False)["label"].mean()


def build_dataset(ratings: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=(ratings.label.min(), ratings.label.max()))
    return Dataset.load_from_df(ratings[["user_id", "item_id", "label"]], reader)



def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def metric_dict(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": mse ** 0.5,
        "MSE": mse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


class LGBMWrapper(AlgoBase):
    def __init__(self, params: Dict | None = None, num_boost_round: int = 100):
        super().__init__()
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}

    def _build_feature(self, u, i, trainset):
        """Basic 6‑dim collaborative feature vector."""
        if u not in self.user_map or i not in self.item_map:
            user_idx = -1
            item_idx = -1
        else:
            user_idx = self.user_map[u]
            item_idx = self.item_map[i]

        user_ratings = [r for (_, r) in trainset.ur.get(u, [])]
        user_mean = np.mean(user_ratings) if user_ratings else trainset.global_mean
        user_count = len(user_ratings)

        item_ratings = [r for (_, r) in trainset.ir.get(i, [])]
        item_mean = np.mean(item_ratings) if item_ratings else trainset.global_mean
        item_count = len(item_ratings)

        return [
            user_idx,
            item_idx,
            user_mean,
            item_mean,
            user_count,
            item_count,
        ]

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # mapping
        self.user_map = {u: idx for idx, u in enumerate(trainset.all_users())}
        self.item_map = {i: idx for idx, i in enumerate(trainset.all_items())}

        X = []
        y = []
        for u, i, r in trainset.all_ratings():
            X.append(self._build_feature(u, i, trainset))
            y.append(r)
        X = np.asarray(X)
        y = np.asarray(y)

        dtrain = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def estimate(self, u, i):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        x = np.asarray([self._build_feature(u, i, self.trainset)])
        return float(self.model.predict(x)[0])


class BiasedMF:
    def __init__(self, n_factors: int = 100, reg: float = 0.02, n_iter: int = 20, lr: float = 0.005, seed: int = SEED):
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter
        self.lr = lr
        self.seed = seed

    def fit(self, trainset):
        self.trainset = trainset
        rng = np.random.default_rng(self.seed)
        n_users = trainset.n_users
        n_items = trainset.n_items
        self.global_mean = trainset.global_mean
        P = 0.1 * rng.standard_normal((n_users, self.n_factors))
        Q = 0.1 * rng.standard_normal((n_items, self.n_factors))
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)

        for _ in range(self.n_iter):
            for u, i, r in trainset.all_ratings():
                err = r - (self.global_mean + bu[u] + bi[i] + np.dot(P[u], Q[i]))
                bu[u] += self.lr * (err - self.reg * bu[u])
                bi[i] += self.lr * (err - self.reg * bi[i])
                P[u] += self.lr * (err * Q[i] - self.reg * P[u])
                Q[i] += self.lr * (err * P[u] - self.reg * Q[i])

        self.P, self.Q, self.bu, self.bi = P, Q, bu, bi
        return self

    def predict(self, uid, iid):
        u_known = uid in self.trainset._raw2inner_id_users
        i_known = iid in self.trainset._raw2inner_id_items
        if not u_known or not i_known:
            est = self.global_mean
        else:
            u = self.trainset.to_inner_uid(uid)
            i = self.trainset.to_inner_iid(iid)
            est = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
        return est


def build_svd_bag(best_params: Dict[str, Any], n_bags: int = 3, factor_jitter: int = 20) -> List[SVD]:
    seeds = [7, 11, 2025][:n_bags]
    bag = []
    for s in seeds:
        p = best_params.copy()
        base = p.get("n_factors", 150)
        p["n_factors"] = max(20, base + random.randint(-factor_jitter // 2, factor_jitter // 2))
        p["random_state"] = s
        bag.append(SVD(**p))
    return bag


def get_out_of_fold_predictions(data: Dataset, base_models: List[Any], n_splits: int = 5):
    """Return numpy arrays X_meta (n_samples × n_models) and y_meta."""
    kf = KFold(n_splits=n_splits, random_state=SEED, shuffle=True)
    fold_preds: List[np.ndarray] = []
    fold_truth: List[np.ndarray] = []

    for trainset, valset in kf.split(data):
        for m in base_models:
            m.fit(trainset)

        X_val = []
        y_val = []
        for uid, iid, r in valset:
            y_val.append(r)
            preds = []
            for m in base_models:
                if isinstance(m, BiasedMF):
                    preds.append(m.predict(uid, iid))
                else:
                    preds.append(m.predict(uid, iid).est)
            X_val.append(preds)
        fold_preds.append(np.vstack(X_val))
        fold_truth.append(np.asarray(y_val))

    X_meta = np.vstack(fold_preds)
    y_meta = np.hstack(fold_truth)
    return X_meta, y_meta


def nnls_blend(X_meta: np.ndarray, y_meta: np.ndarray) -> np.ndarray:
    w, _ = nnls(X_meta, y_meta)
    w /= w.sum()
    return w


def main():
    ap = argparse.ArgumentParser(description="Advanced ensemble recommender v3.1‑debug")
    ap.add_argument("--data_dir", type=Path, default=Path("./dataset/"))
    ap.add_argument("--lgbm_rounds", type=int, default=200, help="Number of boosting rounds for LightGBM")
    args = ap.parse_args()

    ratings, test_df = load_data(args.data_dir)
    ratings = clean_ratings(ratings)
    data = build_dataset(ratings)
    rmin, rmax = ratings.label.min(), ratings.label.max()

    svd_grid = {
        "n_factors": [120, 160],
        "lr_all": [0.005],
        "reg_all": [0.02],
        "n_epochs": [80],
    }
    svdpp_grid = {
        "n_factors": [120],
        "lr_all": [0.005],
        "reg_all": [0.01, 0.015],
        "n_epochs": [100, 120],
    }
    knn_grid = {
        "k": [60],
        "min_k": [3],
        "sim_options": {"name": ["pearson"], "shrinkage": [100], "user_based": [False]},
    }

    def tune_algo(algo_cls, grid, name):
        gs = GridSearchCV(algo_cls, grid, measures=["rmse"], cv=3, n_jobs=-1, joblib_verbose=0)
        gs.fit(data)
        best = algo_cls(**gs.best_params["rmse"])
        return best, gs.best_params["rmse"], gs.best_score["rmse"]

    print("Searching SVD/SVD++/KNN hyper‑params…")
    svd_best, svd_params, _ = tune_algo(SVD, svd_grid, "SVD")
    svdpp_best, _, _ = tune_algo(SVDpp, svdpp_grid, "SVD++")
    knn_best, _, _ = tune_algo(KNNBaseline, knn_grid, "KNN‑Baseline")

    svd_bag = build_svd_bag(svd_params)

    mf_model = BiasedMF(n_factors=150, reg=0.02, n_iter=20, lr=0.005)

    lgbm_model = LGBMWrapper(num_boost_round=args.lgbm_rounds)

    base_models = [svd_best, svdpp_best, knn_best] + svd_bag + [mf_model, lgbm_model]

    print("Generating out‑of‑fold predictions…")
    X_meta, y_meta = get_out_of_fold_predictions(data, base_models)
    weights = nnls_blend(X_meta, y_meta)
    print("Blender NNLS weights:", np.round(weights, 3))

    full_trainset = data.build_full_trainset()
    for m in base_models:
        m.fit(full_trainset)

    print("Predicting test set…")
    n_models = len(base_models)
    n_pairs = len(test_df)
    pred_matrix = np.empty((n_models, n_pairs), dtype=np.float32)

    for idx, m in enumerate(base_models):
        if isinstance(m, BiasedMF):
            pred_matrix[idx] = [m.predict(u, i) for u, i in test_df[["user_id", "item_id"]].values]
        else:
            pred_matrix[idx] = [m.predict(u, i).est for u, i in test_df[["user_id", "item_id"]].values]

    ens_preds = weights @ pred_matrix 
    ens_preds = np.clip(ens_preds, rmin, rmax)

    submission = pd.DataFrame({"id": test_df["id"], "label": ens_preds})
    sub_path = args.data_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print("Saved submission to", sub_path.resolve())


if __name__ == "__main__":
    main()
