from typing import Hashable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.sparse import csr_matrix
from rectools import Columns, ExternalIds, InternalIds
from rectools.dataset import Dataset, Interactions
from rectools.models.base import ModelBase

Scores = Union[Sequence[float], np.ndarray]

class UsersCoverageModel(ModelBase):
    def __init__(self, users_percentage: float, k_recom: int = 10) -> None:
        super().__init__()
        self.users_percentage = users_percentage
        self.k_recom = k_recom
        

    def _get_top_items_covered_users(self, matrix: csr_matrix):
        assert matrix.format == 'csr'
        n_users = int(matrix.shape[0] * self.users_percentage)
        print(f'n_users: {n_users} ({100 * n_users / matrix.shape[0] :.2f}%)')
        
        item_set = []
        covered_users = np.zeros(matrix.shape[0], dtype=bool) # true if a user has been checked already
        while (covered_users.sum() < n_users) or (len(item_set) < self.k_recom): # stop if the number of checked users exceeds the limit
            top_item = mode(matrix[~covered_users].indices)[0][0] # most frequent item among yet unchecked users 
            item_set.append(top_item)
            covered_users += np.maximum.reduceat(matrix.indices==top_item, matrix.indptr[:-1], dtype=bool) 
        return item_set, covered_users

        
    def _fit(self, dataset: Dataset):
        matrix = dataset.get_user_item_matrix()
        self.pop_covered, covered_users = self._get_top_items_covered_users(matrix)
        return self
    

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: Optional[np.ndarray],
        **kwargs
    ) -> Tuple[InternalIds, InternalIds, Scores]:

        # recom = pd.DataFrame()
        reco_user_ids = []
        reco_item_ids= []
        reco_scores = []
        
        for u in user_ids:
            reco_user_ids.extend([u]*k)
            reco_item_ids.extend(list(self.pop_covered[:k]))
            reco_scores.extend(list(np.arange(k, 0, -1)))
        return reco_user_ids, reco_item_ids, reco_scores 
