import os
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from catboost import CatBoostClassifier

from src.models.base import BaseModel


class BaseRanker(BaseModel, ABC):
    def __init__(self, ranker_name: str):
        self.ranker_name = ranker_name

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def recommend_items(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls):
        pass


class TwoStageRanker(BaseRanker):
    """
    Двухэтапный рекоммендер: кандидатогенерация -> ранжирование CatBoost.
 
    ranker_name: str  – название модели
    n_candidates: int – количество кандидатов от каждого рекоммендера
    """
 
    def __init__(
        self,
        ranker_name: str,
        n_candidates: int = 50,
    ):
        super().__init__(ranker_name)
 
        self.n_candidates       = n_candidates
        self.recommenders       = None   # dict[str, RecommenderProtocol]
        self.counter_names      = None   # list[str]
        self.counters_path      = None   # str
        self.counters_df        = None   # pd.DataFrame, прочитанный один раз
        self.track_length_df    = None   # pd.DataFrame с ['item_id', 'track_length_seconds']
        self.user_embeddings_df = None   # pd.DataFrame, index=uid,     cols=emb_names
        self.item_embeddings_df = None   # pd.DataFrame, index=item_id, cols=emb_names
        self.embedding_names    = None   # list[str] — пересечение колонок user/item
        self.model              = None   # CatBoostClassifier
        self.feature_names      = None   # list[str]
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _generate_candidates(self, user_ids: list) -> pd.DataFrame:
        """
        Запрашивает кандидатов у каждого рекоммендера и объединяет их.
        Возвращает DataFrame с колонками ['uid', 'item_id'].
        """
        frames = []
        for name, recommender in self.recommenders.items():
            print(f"  Generating candidates from '{name}'...")
            all_recs = recommender.recommend_items(user_ids, N=self.n_candidates)
            for uid, items in zip(user_ids, all_recs):
                frames.append(pd.DataFrame({'uid': uid, 'item_id': items}))
 
        candidates = pd.concat(frames, ignore_index=True).drop_duplicates(['uid', 'item_id'])
        print(f"  Total unique candidates: {len(candidates):,}")
        return candidates
 
    def _add_counter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Джойнит айтемные счётчики по item_id из self.counters_df."""
        if self.counters_df is None:
            return df
        return df.merge(self.counters_df, on='item_id', how='left')
 
    def _add_track_length_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Джойнит track_length_seconds по item_id из self.track_length_df."""
        if self.track_length_df is None:
            return df
        return df.merge(
            self.track_length_df[['item_id', 'track_length_seconds']],
            on='item_id',
            how='left'
        )
 
    def _add_dot_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Для каждого названия из embedding_names считает dot product
        между user-эмбеддингом и item-эмбеддингом и добавляет колонку '{name}_dot_product'.
        """
        if not self.embedding_names:
            return df
 
        user_emb = self.user_embeddings_df[self.embedding_names]
        item_emb = self.item_embeddings_df[self.embedding_names]
 
        df = df.merge(user_emb.add_prefix('_u_'), left_on='uid',     right_index=True, how='left')
        df = df.merge(item_emb.add_prefix('_i_'), left_on='item_id', right_index=True, how='left')
 
        for name in self.embedding_names:
            u_col = f'_u_{name}'
            i_col = f'_i_{name}'
            u_vecs = np.vstack(df[u_col].values)
            i_vecs = np.vstack(df[i_col].values)
            df[f'{name}_dot_product'] = (u_vecs * i_vecs).sum(axis=1)
            df.drop(columns=[u_col, i_col], inplace=True)
 
        return df
 
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_counter_features(df)
        df = self._add_track_length_feature(df)
        df = self._add_dot_product_features(df)
        return df
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def fit(
        self,
        recommenders: dict,
        catboost_model: CatBoostClassifier,
        feature_names: list[str],
        counter_names: list[str]           = None,
        counters_path: str                 = None,
        user_embeddings_df: pd.DataFrame   = None,
        item_embeddings_df: pd.DataFrame   = None,
        track_length_df: pd.DataFrame      = None,
    ):
        """
        Инициализирует рекоммендер с уже обученными компонентами.
 
        Args:
            recommenders: dict                  – {name: recommender} для кандидатогенерации
            catboost_model: CatBoostClassifier  – уже обученная модель ранжирования
            feature_names: list[str]            – список фичей, на которых обучен catboost_model
            counter_names: list[str]            – названия айтемных счётчиков
            counters_path: str                  – путь к parquet-файлу со счётчиками
            user_embeddings_df: pd.DataFrame    – эмбеддинги юзеров (index=uid, cols=emb_names)
            item_embeddings_df: pd.DataFrame    – эмбеддинги айтемов (index=item_id, cols=emb_names)
            track_length_df: pd.DataFrame       – датафрейм с колонками ['item_id', 'track_length_seconds']
        """
        self.recommenders       = recommenders
        self.model              = catboost_model
        self.feature_names      = feature_names
        self.counter_names      = counter_names or []
        self.counters_path      = counters_path
        self.user_embeddings_df = user_embeddings_df
        self.item_embeddings_df = item_embeddings_df
        self.track_length_df    = track_length_df
 
        # Читаем счётчики один раз
        if self.counter_names and self.counters_path:
            print("Reading counters from parquet...")
            self.counters_df = pd.read_parquet(self.counters_path, columns=self.counter_names)
            print(f"Counters loaded: {self.counter_names}")
        else:
            self.counters_df = None
 
        # Определяем общие названия эмбеддингов
        if user_embeddings_df is not None and item_embeddings_df is not None:
            self.embedding_names = list(
                set(user_embeddings_df.columns) & set(item_embeddings_df.columns)
            )
            print(f"Embedding dot products to compute: {self.embedding_names}")
        else:
            self.embedding_names = []
 
        print("TwoStageRanker is ready!")
 
    def recommend_items(
        self,
        user_id: list,
        N: int = 10,
        user_col: str = 'uid',
        item_col: str = 'item_id',
    ) -> list[list]:
        """
        Генерирует кандидатов, строит фичи, ранжирует CatBoost и возвращает топ-N.
 
        Args:
            user_ids: list – список uid для которых нужны рекомендации
            N: int         – финальное число рекомендаций на юзера
 
        Returns:
            list[list] – список списков item_id в том же порядке что и user_ids
        """
        if self.model is None:
            raise ValueError("Model is not set. Call .fit() first.")
        if self.recommenders is None:
            raise ValueError("Recommenders are not set. Call .fit() first.")
 
        print("Generating candidates...")
        candidates = self._generate_candidates(user_id)
 
        print("Building features...")
        candidates = self._build_features(candidates)
 
        candidates['score'] = self.model.predict_proba(candidates[self.feature_names])[:, 1]
 
        recommendations = (
            candidates
            .sort_values([user_col, 'score'], ascending=[True, False])
            .groupby(user_col)[item_col]
            .apply(lambda x: x.head(N).tolist())
            .to_dict()
        )
 
        # Возвращаем в том же порядке что и user_ids, для неизвестных — []
        return [recommendations.get(uid, []) for uid in user_id]
 
    def save_model(self, directory: str = "."):
        """
        Сохраняет модель и все поля в {directory}/{ranker_name}.joblib
        """
        if self.model is None:
            raise ValueError("Model is not set. Call .fit() first.")
 
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{self.ranker_name}.joblib")
 
        model_state = {
            'ranker_name':        self.ranker_name,
            'n_candidates':       self.n_candidates,
            'counter_names':      self.counter_names,
            'counters_path':      self.counters_path,
            'counters_df':        self.counters_df,
            'track_length_df':    self.track_length_df,
            'user_embeddings_df': self.user_embeddings_df,
            'item_embeddings_df': self.item_embeddings_df,
            'embedding_names':    self.embedding_names,
            'feature_names':      self.feature_names,
            'model':              self.model,
        }
 
        print(f"Saving TwoStageRanker to {model_path}...")
        joblib.dump(model_state, model_path)
        print("TwoStageRanker saved successfully!")
 
    @classmethod
    def from_pretrained(cls, model_path: str, recommenders: dict = None) -> 'TwoStageRanker':
        """
        Загружает модель из сохранённого состояния.
 
        Args:
            model_path: str    – путь к .joblib файлу
            recommenders: dict – {name: recommender} (не сохраняются в joblib, передаются отдельно)
        """
        print(f"Loading TwoStageRanker from {model_path}...")
        state = joblib.load(model_path)
 
        instance = cls(
            ranker_name  = state['ranker_name'],
            n_candidates = state['n_candidates'],
        )
        instance.counter_names      = state['counter_names']
        instance.counters_path      = state['counters_path']
        instance.counters_df        = state['counters_df']
        instance.track_length_df    = state['track_length_df']
        instance.user_embeddings_df = state['user_embeddings_df']
        instance.item_embeddings_df = state['item_embeddings_df']
        instance.embedding_names    = state['embedding_names']
        instance.feature_names      = state['feature_names']
        instance.model              = state['model']
        instance.recommenders       = recommenders
 
        print("TwoStageRanker loaded successfully!")
        return instance