from implicit.als import AlternatingLeastSquares
from numpy import ndarray

class ALS(AlternatingLeastSquares):
    def fit(self, user_items, show_progress=True):
        '''
        Parameters
        ----------
        user_items: csr_matrix
            Matrix of confidences for the liked items. 
            This matrix should be a csr_matrix 
            where the rows of the matrix are the users, 
            the columns are the items that was liked by users, 
            and the value is the confidence that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        '''
        return super().fit(user_items.T, show_progress)

    def recommend(
        self, 
        user_items,
        N=10,
        recalculate_user=False,
        filter_already_liked_items=True,
        filter_items=None,
        num_threads=0,
        show_progress=True,
        batch_size=0,
        users_items_offset=0,
    ) -> ndarray:
    
        return self.recommend_all(
            user_items,
            N,
            recalculate_user,
            filter_already_liked_items,
            filter_items,
            num_threads,
            show_progress,
            batch_size,
            users_items_offset,
        )