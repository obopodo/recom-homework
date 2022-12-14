from typing import Optional, Sequence, Tuple
from lightfm import LightFM
from rectools.models.base import ModelBase
from rectools.dataset import Dataset

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

UserIDs = Sequence[int]
ItemIDs = Sequence[int]
Scores = Sequence[float]


class LightFMCustom(LightFM, ModelBase):
    def __init__(self, 
        no_components=10, 
        k=5, 
        n=10, 
        learning_schedule="adagrad", 
        loss="logistic", 
        learning_rate=0.05, 
        rho=0.95, 
        epsilon=0.000001, 
        item_alpha=0, 
        user_alpha=0, 
        max_sampled=10, 
        random_state=None,
        verbose=False
    ):
        '''
        Rectools-style wrapper for LightFM 
        '''
        super().__init__(
            no_components, 
            k, 
            n, 
            learning_schedule, 
            loss, 
            learning_rate, 
            rho, 
            epsilon, 
            item_alpha, 
            user_alpha, 
            max_sampled, 
            random_state)
        self.is_fitted = False
        self.verbose = verbose

    def fit(self, 
        dataset: Dataset, 
        epochs=1, 
        num_threads=1,
    ):
        interactions_matrix = dataset.get_user_item_matrix()
        super().fit(
            interactions=interactions_matrix.sign(), 
            # user_features | None = None, # TODO implement the features conversion from Rectools to LightFM format
            # item_features | None = None, 
            sample_weight=interactions_matrix.tocoo(), 
            epochs=epochs, 
            num_threads=num_threads, 
            verbose=self.verbose,
        )
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

        ones_items_like = np.ones_like(internal_item_ids)
        predicted_weights = np.vstack(
            [
                self.predict(
                    user_ids=ones_items_like * user_id, 
                    item_ids=internal_item_ids,
                    # item_features, # TODO implement the features conversion from Rectools to LightFM format
                    # user_features, 
                    # num_threads,
                ) for user_id in tqdm(user_ids, desc='Recoms for users')
            ]
        )
        
        # argpartition is faster than sort and we don't need to sort unused items
        top_items = np.argpartition(predicted_weights, -k, axis=1)[:, -k:] # k last indexes are top items (internal)

        # now let's sort only top items using weights
        range_index = np.arange(len(predicted_weights))[:, None] # indexes to obtain rows
        top_weights = predicted_weights[range_index, top_items] # get top weights for every row
        top_weight_argsorted = np.argsort(top_weights, axis=1)
        top_weights_sorted = top_weights[range_index, top_weight_argsorted]
        top_items_sorted = top_items[range_index, top_weight_argsorted]

        reco_user_ids = user_ids[:, None] @ np.ones((1, k)) # one row - one user id repeated k times

        return reco_user_ids.flatten(), top_items_sorted.flatten(), top_weights_sorted.flatten()
