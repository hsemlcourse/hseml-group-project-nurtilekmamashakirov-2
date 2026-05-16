from typing import Literal
import pandas as pd
import scipy.sparse as sparse
import numpy as np
import implicit
import os
import joblib
from abc import ABC, abstractclassmethod, abstractstaticmethod
from src.models.base import BaseModel

class BaseMatrixFactorization(BaseModel, ABC):
    """
    Базовый класс для матричных разложений

    factorization_name: str – название разложения
    """
    def __init__(self, factorization_name: str):
        self.factorization_name = factorization_name

    @abstractclassmethod
    def get_embeddings(self):
        pass

    @abstractclassmethod
    def save_embeddings(self):
        pass

    @abstractclassmethod
    def save_model(self):
        pass

    @abstractstaticmethod
    def from_pretrained(cls):
        pass
        

class ALS(BaseMatrixFactorization):
    """
    Класс для ALS-разложений

    factors: int – размер эмбеддингов
    regularization: float – коэффициент регуляризации
    iterations: int – кол-во ALS-шагов
    random_state: int – 
    """
    def __init__(self, factorization_name: str, factors: int = 64, regularization: float = 0.01,
                 iterations: int = 20, random_state: int = 42):
        super().__init__(factorization_name)

        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state
        )

        self.user_to_idx = None
        self.item_to_idx = None
        self.user_original_ids = None
        self.item_original_ids = None
        self.user_embeddings_df = None
        self.item_embeddings_df = None
        self.user_item_matrix = None

        # Set environment variable to avoid OpenBLAS performance issues
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

    def _prepare_data(self, df: pd.DataFrame, 
                     user_col: str = 'uid', 
                     item_col: str = 'item_id',
                     event_type_col: str = 'event_type',
                     event_weights: dict = None, 
                     listen_weight: float = 0.01):
        """
        Prepares the data for ALS training by creating a sparse user-item matrix
        and ID mappings, applying specified weights based on event_type.

        Подготовка разряженной матрицы для обучения ALS-разложения на заданный таргет
        Производится маппинг id -> index, index -> id для user'ов и item'ов

        df: pd.DataFrame – датафрейм с интеракциями
        user_col: str – название колонки с id user'а
        item_col: str – название колонки с id item'а
        event_type_col: str – название колонки с типом взаимодействия
        event_weights: dict – вес каждого ивента в итоговую матрицу
        listen_weight: float – вес длителньости прослушивания
        """
        # Create mappings for user and item IDs to contiguous integers
        user_ids_category = df[user_col].astype("category")
        item_ids_category = df[item_col].astype("category")

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids_category.cat.categories)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids_category.cat.categories)}

        self.user_original_ids = user_ids_category.cat.categories.to_list()
        self.item_original_ids = item_ids_category.cat.categories.to_list()


        # Initialize data array with default value (e.g., 0.0 for interactions not explicitly handled)
        data = pd.Series([0.0] * len(df), index=df.index)

        # Apply listen_weight for 'listen' events
        listen_mask = df[event_type_col] == 'listen'
        data.loc[listen_mask] = df.loc[listen_mask, 'played_ratio_pct'] * listen_weight

        # Apply event_weights for other specific event types
        for event, weight in event_weights.items():
            event_mask = df[event_type_col] == event
            data.loc[event_mask] = weight

        rows = user_ids_category.cat.codes  # Mapped user indices
        cols = item_ids_category.cat.codes  # Mapped item indices

        # Create the user-item sparse matrix (CSR format for efficient row slicing)
        user_item_matrix = sparse.csr_matrix(
            (data.values, (rows, cols)),
            shape=(len(self.user_original_ids), len(self.item_original_ids))
        )
        return user_item_matrix

    def fit(self, 
            df: pd.DataFrame, 
            user_col: str = 'uid', item_col: str = 'item_id',
            event_type_col: str = 'event_type',
            event_weights: dict = {"likes": 1.0, "dislikes": -1.0, "unlikes": -1.0, "undislikes": -1.0}, 
            listen_weight: float = 0.01):
        """
        Обучение ALS-модели
        """
        print("Preparing data for ALS model training with custom event weights...")
        self.user_item_matrix = self._prepare_data(df, user_col, item_col,
                                             event_type_col=event_type_col,
                                             event_weights=event_weights,
                                             listen_weight=listen_weight)

        # Train on user-item matrix directly (users x items)
        print("Training ALS model...")
        self.model.fit(self.user_item_matrix)
        print("ALS model trained successfully!")

    # def recommend_items(self, user_id: int | list, N: int = 10) -> pd.DataFrame:
    #     """
    #     Рекоммендация N айтемов из разложения
    #     """
    #     if self.model is None:
    #         raise ValueError("Model has not been trained. Call .fit() first.")
    #     if self.user_to_idx is None or self.item_original_ids is None:
    #         raise ValueError("Data mappings (user_to_idx, item_original_ids) are not available. Call .fit() first.")

    #     if user_id not in self.user_to_idx:
    #         print(f"Warning: User ID {user_id} not found in training data mappings. Cannot provide recommendations.")
    #         return pd.DataFrame(columns=['item_id', 'score'])

    #     user_idx = self.user_to_idx[user_id]

    #     # Explicitly get the user's interaction row from the user-item matrix
    #     # The 'implicit' library's recommend method expects the specific user's interaction row,
    #     # not the entire user-item matrix, for the 'user_items' parameter when 'userid' is a single integer.
    #     user_interaction_row = self.user_item_matrix[user_idx]

    #     # Use the implicit library's recommend method
    #     recommendations = self.model.recommend(
    #         user_idx,
    #         user_interaction_row, # Pass only the specific user's interaction row
    #         N=N,
    #         filter_already_liked_items=True
    #     )

    #     # recommendations is a tuple of (item_indices, scores)
    #     recommended_item_indices = recommendations[0]
    #     scores = recommendations[1]

    #     # Convert internal item indices back to original item_ids
    #     recommended_item_ids = [self.item_original_ids[idx] for idx in recommended_item_indices]

    #     return pd.DataFrame({'item_id': recommended_item_ids, 'score': scores})

    def recommend_items(self, user_id: int | list, N: int = 10) -> pd.DataFrame | list[list]:
        """
        Рекоммендация N айтемов из разложения.

        Args:
            user_id: int | list – id одного юзера или список id юзеров
            N: int – количество рекомендаций

        Returns:
            pd.DataFrame – если user_id: int, датафрейм с колонками ['item_id', 'score']
            list[list]   – если user_id: list, список списков item_id в том же порядке что и user_id
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call .fit() first.")
        if self.user_to_idx is None or self.item_original_ids is None:
            raise ValueError("Data mappings are not available. Call .fit() first.")

        # --- Одиночный запрос ---
        if isinstance(user_id, int):
            if user_id not in self.user_to_idx:
                raise KeyError(f"User ID {user_id} not found in training data.")

            user_idx = self.user_to_idx[user_id]
            item_indices, scores = self.model.recommend(
                user_idx,
                self.user_item_matrix[user_idx],
                N=N,
                filter_already_liked_items=True
            )
            recommended_item_ids = [self.item_original_ids[idx] for idx in item_indices]
            return pd.DataFrame({'item_id': recommended_item_ids, 'score': scores})

        # --- Батчевый запрос ---
        unknown = [uid for uid in user_id if uid not in self.user_to_idx]
        if unknown:
            raise KeyError(f"User ID(s) not found in training data: {unknown}")

        user_indices = np.array([self.user_to_idx[uid] for uid in user_id])
        batch_item_indices, _ = self.model.recommend(
            user_indices,
            self.user_item_matrix[user_indices],
            N=N,
            filter_already_liked_items=True
        )

        return [
            [self.item_original_ids[idx] for idx in item_indices]
            for item_indices in batch_item_indices
        ]

    def get_embeddings(self):
        """
        Получение всех user и item эмбеддингов из разложения
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call .fit() first.")

        # User embeddings are from model.user_factors (as model trained on user-item matrix)
        user_embeddings = self.model.user_factors
        user_embeddings_df = pd.DataFrame(user_embeddings, index=self.user_original_ids)
        user_embeddings_df.index.name = 'uid'
        self.user_embeddings_df = user_embeddings_df.apply(lambda x: x.tolist(), axis=1).to_frame(name=self.factorization_name)
        self.user_embeddings_df.index.name = user_embeddings_df.index.name

        # Item embeddings are from model.item_factors (as model trained on user-item matrix)
        item_embeddings = self.model.item_factors
        item_embeddings_df = pd.DataFrame(item_embeddings, index=self.item_original_ids)
        item_embeddings_df.index.name = 'item_id'
        self.item_embeddings_df = item_embeddings_df.apply(lambda x: x.tolist(), axis=1).to_frame(name=self.factorization_name)
        self.item_embeddings_df.index.name = item_embeddings_df.index.name

        return self.user_embeddings_df, self.item_embeddings_df

    def save_embeddings(self, user_embeddings_path: str, item_embeddings_path: str):
        """
        Сохранения эмбеддингов по пути user_embeddings_path и item_embeddings_path
        Если файл существует, то эмбеддинги добавятся в новую колонку с названием factorization_name
        """
        if self.user_embeddings_df is None or self.item_embeddings_df is None:
            user_embeddings, item_embeddings = self.get_embeddings()
        else:
            user_embeddings = self.user_embeddings_df
            item_embeddings = self.item_embeddings_df

        # Save User Embeddings
        print(f"Saving user embeddings to {user_embeddings_path}...")
        if os.path.exists(user_embeddings_path):
            existing_user_embeddings = pd.read_parquet(user_embeddings_path)
            merged_user_embeddings = existing_user_embeddings.merge(
                user_embeddings,
                left_index=True,
                right_index=True,
                how='outer'
            )
            merged_user_embeddings.to_parquet(user_embeddings_path)
        else:
            user_embeddings.to_parquet(user_embeddings_path)
        print(f"User embeddings saved successfully for factorization: {self.factorization_name}")

        # Save Item Embeddings
        print(f"Saving item embeddings to {item_embeddings_path}...")
        if os.path.exists(item_embeddings_path):
            existing_item_embeddings = pd.read_parquet(item_embeddings_path)
            merged_item_embeddings = existing_item_embeddings.merge(
                item_embeddings,
                left_index=True,
                right_index=True,
                how='outer'
            )
            merged_item_embeddings.to_parquet(item_embeddings_path)
        else:
            item_embeddings.to_parquet(item_embeddings_path)
        print(f"Item embeddings saved successfully for factorization: {self.factorization_name}")

    def save_model(self, directory: str = "."):
        """
        Сохранение модели и состояния всех полей по пути directory
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call .fit() first.")

        model_filename = f"{self.factorization_name}.joblib"
        model_path = os.path.join(directory, model_filename)

        # Save all relevant attributes for full model reconstruction
        model_state = {
            'model': self.model,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'user_original_ids': self.user_original_ids,
            'item_original_ids': self.item_original_ids,
            'user_item_matrix': self.user_item_matrix,
            'factorization_name': self.factorization_name,
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'random_state': self.random_state
        }

        os.makedirs(directory, exist_ok=True)
        print(f"Saving ALS model and state to {model_path}...")
        joblib.dump(model_state, model_path)
        print(f"ALS model and state saved successfully as {model_filename}!")

    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Извлечение модели из сохраненного состояния
        """
        print(f"Loading ALS model and state from {model_path}...")
        model_state = joblib.load(model_path)

        # Reconstruct ALSCombiner instance
        instance = cls(
            factorization_name=model_state['factorization_name'],
            factors=model_state['factors'],
            regularization=model_state['regularization'],
            iterations=model_state['iterations'],
            random_state=model_state['random_state']
        )
        instance.model = model_state['model']
        instance.user_to_idx = model_state['user_to_idx']
        instance.item_to_idx = model_state['item_to_idx']
        instance.user_original_ids = model_state['user_original_ids']
        instance.item_original_ids = model_state['item_original_ids']
        instance.user_item_matrix = model_state['user_item_matrix']

        print(f"ALS model and state loaded successfully from {model_path}!")
        return instance
