import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, Iterable, List
from tqdm.auto import tqdm
from rectools.model_selection import TimeRangeSplitter
from rectools.dataset import Interactions, Dataset
from rectools.metrics import calc_metrics
from rectools.metrics.base import MetricAtK
from rectools import Columns
from rectools.models.base import ModelBase

class CrossValidator:
    def __init__(
        self,
        last_date: str,
        n_folds: int = 3,
        n_days_per_fold: int = 7,
        filter_already_seen: bool = True,
        filter_cold_items: bool = True,
        filter_cold_users: bool = True,
    ) -> None:

        unit = "D"
        self.last_date = pd.to_datetime(last_date)

        self.start_date = self.last_date - pd.Timedelta(n_folds * n_days_per_fold - 1, unit=unit)
        periods = n_folds + 1
        freq = f"{n_days_per_fold}{unit}"

        date_range = pd.date_range(start=self.start_date, periods=periods, freq=freq, tz=self.last_date.tz)
        self.cv = TimeRangeSplitter(
            date_range=date_range,
            filter_already_seen=filter_already_seen,
            filter_cold_items=filter_cold_items,
            filter_cold_users=filter_cold_users,
        )

        print((
            f'start_date: {self.start_date}\n'
            f'last_date: {self.last_date}\n'
            f'periods: {periods}\n'
            f'freq: {freq}\n'
        ))


    def validate(
        self, 
        models: Dict[str, ModelBase],
        metrics: Dict[str, MetricAtK],
        interactions: Interactions, 
        k_recos: int = 10,
        item_features: pd.DataFrame = None,
        user_features: pd.DataFrame = None,
        cat_user_features: Iterable = (),
        cat_item_features: Iterable = (),
    ) -> pd.DataFrame:

        fold_iterator = self.cv.split(interactions)
        results =[]

        pbar_folds = tqdm(
            enumerate(fold_iterator), 
            total=self.cv.get_n_splits(interactions), 
            desc='Fold', 
            position=0, 
            leave=False
        )

        for i_fold, (train_ids, test_ids, _) in pbar_folds:
            df_train = interactions.df.iloc[train_ids]
            if item_features is not None:
                item_features_train = item_features[
                    item_features[Columns.Item]
                    .isin(df_train[Columns.Item])
                ]
            else:
                item_features_train = None
            
            if user_features is not None:
                user_features_train = user_features[
                    user_features[Columns.User].isin(df_train[Columns.User])
                ]
            else: 
                user_features_train = None
            
            dataset = Dataset.construct(
                interactions_df=df_train,
                user_features_df=user_features_train,
                item_features_df=item_features_train,
                cat_item_features=cat_item_features,
                cat_user_features=cat_user_features,
            )

            df_test = interactions.df.iloc[test_ids][Columns.UserItem]
            test_users = np.unique(df_test[Columns.User])
            
            pbar_models = tqdm(
                models.items(), 
                desc='Models', 
                position=1, 
                leave=False, 
                total=len(models)
            )
            for model_name, model in pbar_models:
                pbar_models.set_description(f'Models ({model_name})')
                model = deepcopy(model)
                model.fit(dataset)
                recos = model.recommend(
                    users=test_users,
                    dataset=dataset,
                    k=k_recos,
                    filter_viewed=True,
                )
                metric_values = calc_metrics(
                    metrics,
                    reco=recos,
                    interactions=df_test,
                    prev_interactions=df_train
                )
                res = {"fold": i_fold, "model": model_name}
                res.update(metric_values)
                results.append(res)
            
        return pd.DataFrame(results)
