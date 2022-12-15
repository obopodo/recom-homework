import numpy as np
import pandas as pd
from typing import Sequence, Optional, Tuple

from implicit.als import AlternatingLeastSquares
from implicit.cuda import HAS_CUDA
from rectools.models.base import ModelBase
from rectools.dataset import Dataset


UserIDs = Sequence[int]
ItemIDs = Sequence[int]
Scores = Sequence[float]

class ALS(AlternatingLeastSquares, ModelBase):
    def __init__(self, 
        factors=100, 
        regularization=0.01, 
        dtype=np.float32, 
        use_native=True, 
        use_cg=True, 
        use_gpu=HAS_CUDA, 
        iterations=15, 
        calculate_training_loss=False, 
        num_threads=0, 
        random_state=None,
        verbose=True
    ):
        '''
        Rectools-style wrapper for implicit's ALS 
        '''
        super().__init__(factors, 
            regularization, 
            dtype, 
            use_native, 
            use_cg, 
            use_gpu, 
            iterations, 
            calculate_training_loss, 
            num_threads, 
            random_state
        )
        self.is_fitted = False
        self.verbose = verbose


    def fit(self, dataset: Dataset):
        '''
        Parameters
        ----------
        dataset: rectools.dataset.Dataset
        '''
        user_items = dataset.get_user_item_matrix()
        super().fit(user_items.T, show_progress=self.verbose)
        self.is_fitted = True
        return self

    
    def _recommend_u2i(
        self,
        user_ids: np.ndarray, # internal user ids
        dataset: Dataset,
        k: int,
        filter_viewed: bool=True, # left for compatibility with Rectools
        sorted_item_ids_to_recommend: Optional[np.ndarray] = None, # left for compatibility with Rectools
        **kwargs
    ) -> Tuple[UserIDs, ItemIDs, Scores]:

        internal_item_ids = dataset.item_id_map.internal_ids
        interactions_matrix = dataset.get_user_item_matrix()
        recommended_items = self.recommend_all(
            user_items=interactions_matrix,
            N=k,
            recalculate_user=False,
            filter_already_liked_items=filter_viewed,
            show_progress=self.verbose,
            **kwargs
        )[user_ids] # obtain recos only for passed users
        
        reco_user_ids = user_ids[:, None] @ np.ones((1, k)) # one row - one user id repeated k times
        dummy_weights = np.tile(np.arange(10, 0, -1), (len(user_ids), 1)) # reversed rank

        return reco_user_ids.flatten(), recommended_items.flatten(), dummy_weights.flatten()

    
    def recommend(self, 
        users: np.ndarray, 
        dataset: Dataset, 
        k: int, 
        filter_viewed: bool, 
        items_to_recommend: Optional[np.ndarray] = None, 
        add_rank_col: bool = True
    ) -> pd.DataFrame:
        return ModelBase.recommend(self, users, dataset, k, filter_viewed, items_to_recommend, add_rank_col)
