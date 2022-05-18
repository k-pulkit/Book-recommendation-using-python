from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# We will be implementing this class to
class SVD_recommender(object):
    def __init__(self, processed_ratings_matrix, user_index, item_index):
        self.data = processed_ratings_matrix
        self.U = None
        self.Vt = None
        self.sigma = None
        self.sigmaVt_T = None
        self.results = None
        self.user_idx = user_index
        self.item_idx = item_index

    def fit(self, k=10):
        self.U, self.sigma, self.Vt = svds(self.data, k)
        self.sigma = np.diag(self.sigma)
        self.sigmaVt_T = np.dot(self.sigma, self.Vt).T

    def recommend_with_itemid(self, item_id):
        if item_id not in self.item_idx:
            raise KeyError

        idx = np.where(self.item_idx == item_id)[0][0]
        item = [self.sigmaVt_T[idx]]

        scores = cosine_similarity(self.sigmaVt_T, item)[:, 0]
        return pd.DataFrame({"item_id": self.item_idx, "scores": scores}).sort_values("scores", ascending=False).query(
            "item_id != @item", )
