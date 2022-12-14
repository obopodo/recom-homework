from lightfm import LightFM
import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


class LightFMCustom(LightFM):
    def fit(self, 
        interactions: csr_matrix, 
        user_features=None, 
        item_features=None, 
        epochs=1, 
        num_threads=1, 
        verbose=False
    ):
        return super().fit(
            interactions.sign(), 
            user_features, 
            item_features, 
            interactions.tocoo(), 
            epochs, 
            num_threads, 
            verbose
        )

    def recommend(
        self,
        user_ids,
        item_ids,
        N=10,
        item_features=None,
        user_features=None,
        num_threads=1,
    ) -> np.ndarray:
        ones_items_like = np.ones((len(item_ids, )))
        result = np.vstack(
            [
                self.predict(
                    ones_items_like * user_id, 
                    item_ids,
                    item_features, 
                    user_features, 
                    num_threads,
                ) for user_id in tqdm(user_ids, desc='Recoms for users')
            ]
        )
        result = np.argsort(result, axis=1)[:, -1:-N-1:-1]
        return result