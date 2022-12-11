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
    def __init__(self, users_percentage: float) -> None:
        super().__init__()
        self.users_percentage = users_percentage
        

    def _get_top_items_covered_users(self, matrix: csr_matrix):
        assert matrix.format == 'csr'
        n_users = int(matrix.shape[0] * self.users_percentage / 100)
        print(f'n_users: {n_users} ({100 * n_users / matrix.shape[0] :.2f}%)')
        
        item_set = []
        covered_users = np.zeros(matrix.shape[0], dtype=bool) # true if a user has been checked already
        while (covered_users.sum() < n_users): # stop if the number of checked users exceeds the limit
            top_item = mode(matrix[~covered_users].indices)[0][0] # most frequent item among yet unchecked users 
            item_set.append(top_item)
            covered_users += np.maximum.reduceat(matrix.indices==top_item, matrix.indptr[:-1], dtype=bool) 
        return item_set, covered_users

        
    def _fit(self, dataset: Dataset):
        matrix = dataset.get_user_item_matrix()
        item_set, covered_users = self._get_top_items_covered_users(matrix)
        self.pop_covered = dataset.item_id_map.convert_to_external(item_set) 
        return self
    

    # def recommend(
    #     self, 
    #     users: Union[Sequence[Hashable], np.ndarray],
    #     dataset: Dataset, # оставлено для обратной совместимости
    #     k: int,
    #     filter_viewed=True,
    #     **kwargs
    # ) -> pd.DataFrame:
    #     recom = pd.DataFrame()
    #     for u in users:
    #         user_recom = pd.DataFrame({
    #             'user_id': [u]*k, 
    #             'item_id': self.pop_covered[:k], 
    #             'rank': np.arange(1, k+1)
    #         })
    #         recom = pd.concat((recom, user_recom))
    #     return recom

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        # sorted_item_ids_to_recommend: Optional[np.ndarray],
        **kwargs
    ) -> Tuple[InternalIds, InternalIds, Scores]:

        recom = pd.DataFrame()
        for u in user_ids:
            user_recom = pd.DataFrame({
                'user_id': [u]*k, 
                'item_id': self.pop_covered[:k], 
                'rank': np.arange(1, k+1)
            })
            recom = pd.concat((recom, user_recom))
        return recom
