import pandas as pd
from abc import ABC, abstractclassmethod, abstractstaticmethod
import os

from src.models.base import BaseModel


class BaseCounterModel(BaseModel, ABC):
    """
    Базовый класс для рекоммендера, основанного на счетчиках
    """
    def __init__(self, counter_name: str):
        self.counter_name = counter_name

    @abstractclassmethod
    def get_counters(self):
        pass

    @abstractclassmethod
    def save_counters(self):
        pass

    @abstractclassmethod
    def save_model(self):
        pass
    
    @abstractstaticmethod
    def from_pretrained(cls):
        pass


# class EventCounter(BaseCounterModel):
#     """
#     Рекоммендер, основанный на частнотности айтемов по типу события
#     """

#     def __init__(self, counter_name: str):
#         super().__init__(counter_name)
#         self.item_popularity = None

#     def fit(self, 
#             train_df: pd.DataFrame, 
#             target_event_type: str,
#             item_col: str = 'item_id', 
#             event_type_col: str = 'event_type'):
#         """
#         Вычисляет популярность элементов на основе их встречаемости в обучающих данных. 
#         Популярность определяется как количество событий target_event_type для каждого элемента.

#         Args:
#             train_df: pd.DataFrame – Датафрейм для обучения
#             item_col: str – Название колонки с id item'ов
#             event_type_col: str – Название колонки с типами событий
#         """
#         print(f"Fitting TopPop model for event {target_event_type}...")

#         traget_events = train_df[train_df[event_type_col] == target_event_type]
#         self.item_popularity = traget_events[item_col].value_counts().reset_index()
#         self.item_popularity.columns = [item_col, 'popularity_score']
#         self.item_popularity = self.item_popularity.sort_values(by='popularity_score', ascending=False)

#         print("TopPop model fitted successfully!")

#     def recommend_items(self, user_id: int = None, N: int = 10) -> pd.DataFrame:
#         """
#         Рекоммендует N айтемов для user_id (для любого user_id предсказания одинаковые)

#         Args:
#             user_id: int – id user'а
#             N (int): Количество айтемов для рекомендации

#         Returns:
#             pd.DataFrame: A DataFrame with 'item_id' and 'popularity_score' for the top N items.
#         """
#         if self.item_popularity is None:
#             raise ValueError("TopPop model has not been fitted. Call .fit() first.")

#         return self.item_popularity.head(N).copy()

#     def get_top_k_items(self, k: int = 10) -> pd.DataFrame:
#         """
#         Возвращает k самых популярных айтемов
#         """
#         return self.recommend_items(N=k)
    


# class ListeningTimeCounter(BaseCounterModel):
#     """
#     Рекоммендер, основанный на популярности по суммарному времени прослушивания
#     """

#     def __init__(self, counter_name: str):
#         super().__init__(counter_name)
#         self.item_popularity = None

#     def fit(self,
#             train_df: pd.DataFrame,
#             target_event_type: str = '',
#             item_col: str = 'item_id',
#             event_type_col: str = 'event_type',
#             listening_time_col: str = 'played_ratio_pct'):
#         """
#         Вычисляет популярность элементов на основе суммарного времени прослушивания.

#         Args:
#             train_df: pd.DataFrame – Датафрейм для обучения
#             target_event_type: str – Тип события прослушивания (например, 'play')
#             item_col: str – Название колонки с id item'ов
#             event_type_col: str – Название колонки с типами событий
#             listening_time_col: str – Название колонки с временем прослушивания
#         """
#         print(f"Fitting ListeningTimeCounter model...")

#         target_events = train_df[train_df[event_type_col] == target_event_type]
#         self.item_popularity = (
#             target_events
#             .groupby(item_col)[listening_time_col]
#             .sum()
#             .reset_index()
#         )
#         self.item_popularity.columns = [item_col, 'popularity_score']
#         self.item_popularity = self.item_popularity.sort_values(
#             by='popularity_score', ascending=False
#         )

#         print("ListeningTimeCounter model fitted successfully!")

#     def recommend_items(self, user_id: int = None, N: int = 10) -> pd.DataFrame:
#         """
#         Рекоммендует N айтемов для user_id (для любого user_id предсказания одинаковые)

#         Args:
#             user_id: int – id user'а
#             N: int – Количество айтемов для рекомендации

#         Returns:
#             pd.DataFrame: Датафрейм с 'item_id' и 'popularity_score' для топ-N айтемов
#         """
#         if self.item_popularity is None:
#             raise ValueError("ListeningTimeCounter model has not been fitted. Call .fit() first.")

#         return self.item_popularity.head(N).copy()

#     def get_top_k_items(self, k: int = 10) -> pd.DataFrame:
#         """
#         Возвращает k самых популярных айтемов по суммарному времени прослушивания
#         """
#         return self.recommend_items(N=k)
    
class EventCounter(BaseCounterModel):
    """
    Рекоммендер, основанный на частотности айтемов по типу события
    """
 
    def __init__(self, counter_name: str):
        super().__init__(counter_name)
        self.item_popularity = None
        self.target_event_type = None
 
    def fit(self,
            train_df: pd.DataFrame,
            target_event_type: str,
            item_col: str = 'item_id',
            event_type_col: str = 'event_type'):
        """
        Вычисляет популярность элементов на основе их встречаемости в обучающих данных.
        Популярность определяется как количество событий target_event_type для каждого элемента.
 
        Args:
            train_df: pd.DataFrame – Датафрейм для обучения
            target_event_type: str – Тип события, по которому считается популярность
            item_col: str – Название колонки с id item'ов
            event_type_col: str – Название колонки с типами событий
        """
        print(f"Fitting EventCounter model for event '{target_event_type}'...")
 
        self.target_event_type = target_event_type
 
        target_events = train_df[train_df[event_type_col] == target_event_type]
        self.item_popularity = target_events[item_col].value_counts().reset_index()
        self.item_popularity.columns = [item_col, 'popularity_score']
        self.item_popularity = self.item_popularity.sort_values(
            by='popularity_score', ascending=False
        ).reset_index(drop=True)
 
        print("EventCounter model fitted successfully!")
 
    def recommend_items_df(self, user_id: int = None, N: int = 10) -> pd.DataFrame:
        """
        Рекоммендует N айтемов (для любого user_id предсказания одинаковые)
 
        Args:
            user_id: int – id user'а (не используется, только для совместимости)
            N: int – Количество айтемов для рекомендации
 
        Returns:
            pd.DataFrame: Датафрейм с 'item_id' и 'popularity_score' для топ-N айтемов
        """
        if self.item_popularity is None:
            raise ValueError("EventCounter model has not been fitted. Call .fit() first.")
 
        return self.item_popularity.head(N).copy()
 
    def recommend_items(self, user_id: int | list = None, N: int = 10) -> list:
        """
        Возвращает N самых популярных айтемов в виде листа
        """
        items = list(self.recommend_items_df(N=N)['item_id'])

        if isinstance(user_id, list):
            return [items for _ in range(len(user_id))]
        else:
            return items
 
    def get_counters(self) -> pd.DataFrame:
        """
        Возвращает таблицу популярности всех айтемов
        """
        if self.item_popularity is None:
            raise ValueError("EventCounter model has not been fitted. Call .fit() first.")
        return self.item_popularity.copy()
 
    def save_counters(self, path: str):
        """
        Сохраняет счётчики популярности в parquet-файл.
        Если файл существует — добавляет колонку с именем counter_name.
 
        Args:
            path: str – Путь к parquet-файлу
        """
        counters = self.get_counters().set_index('item_id').rename(
            columns={'popularity_score': self.counter_name}
        )
 
        print(f"Saving EventCounter counters to {path}...")
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            merged = existing.merge(counters, left_index=True, right_index=True, how='outer')
            merged.to_parquet(path)
        else:
            counters.to_parquet(path)
        print(f"EventCounter counters saved successfully as column '{self.counter_name}'!")
 
    def save_model(self, directory: str = "."):
        """
        Сохраняет модель и все поля через joblib по пути directory/{counter_name}.joblib
 
        Args:
            directory: str – Директория для сохранения
        """
        if self.item_popularity is None:
            raise ValueError("EventCounter model has not been fitted. Call .fit() first.")
 
        model_state = {
            'counter_name': self.counter_name,
            'target_event_type': self.target_event_type,
            'item_popularity': self.item_popularity,
        }
 
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{self.counter_name}.joblib")
        print(f"Saving EventCounter model to {model_path}...")
        joblib.dump(model_state, model_path)
        print(f"EventCounter model saved successfully!")
 
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'EventCounter':
        """
        Загружает модель из сохранённого состояния.
 
        Args:
            model_path: str – Путь к .joblib файлу
 
        Returns:
            EventCounter: Восстановленный экземпляр модели
        """
        print(f"Loading EventCounter model from {model_path}...")
        model_state = joblib.load(model_path)
 
        instance = cls(counter_name=model_state['counter_name'])
        instance.target_event_type = model_state['target_event_type']
        instance.item_popularity = model_state['item_popularity']
 
        print(f"EventCounter model loaded successfully from {model_path}!")
        return instance
    

class ListeningTimeCounter(BaseCounterModel):
    """
    Рекоммендер, основанный на популярности по суммарному времени прослушивания
    """
 
    def __init__(self, counter_name: str):
        super().__init__(counter_name)
        self.item_popularity = None
        self.target_event_type = None
        self.listening_time_col = None
 
    def fit(self,
            train_df: pd.DataFrame,
            target_event_type: str = 'listen',
            item_col: str = 'item_id',
            event_type_col: str = 'event_type',
            listening_time_col: str = 'played_ratio_pct'):
        """
        Вычисляет популярность элементов на основе суммарного времени прослушивания.
 
        Args:
            train_df: pd.DataFrame – Датафрейм для обучения
            target_event_type: str – Тип события прослушивания (например, 'listen')
            item_col: str – Название колонки с id item'ов
            event_type_col: str – Название колонки с типами событий
            listening_time_col: str – Название колонки с временем прослушивания
        """
        print(f"Fitting ListeningTimeCounter model for event '{target_event_type}'...")
 
        self.target_event_type = target_event_type
        self.listening_time_col = listening_time_col
 
        target_events = train_df[train_df[event_type_col] == target_event_type]
        self.item_popularity = (
            target_events
            .groupby(item_col)[listening_time_col]
            .sum()
            .reset_index()
        )
        self.item_popularity.columns = [item_col, 'popularity_score']
        self.item_popularity = self.item_popularity.sort_values(
            by='popularity_score', ascending=False
        ).reset_index(drop=True)
 
        print("ListeningTimeCounter model fitted successfully!")
 
    def recommend_items_df(self, user_id: int = None, N: int = 10) -> pd.DataFrame:
        """
        Рекоммендует N айтемов (для любого user_id предсказания одинаковые)
 
        Args:
            user_id: int – id user'а (не используется, только для совместимости)
            N: int – Количество айтемов для рекомендации
 
        Returns:
            pd.DataFrame: Датафрейм с 'item_id' и 'popularity_score' для топ-N айтемов
        """
        if self.item_popularity is None:
            raise ValueError("ListeningTimeCounter model has not been fitted. Call .fit() first.")
 
        return self.item_popularity.head(N).copy()
 
    def recommend_items(self, user_id: int | list = None, N: int = 10) -> list:
        """
        Возвращает N самых популярных айтемов в виде листа
        """
        items = list(self.recommend_items_df(N=N)['item_id'])

        if isinstance(user_id, list):
            return [items for _ in range(len(user_id))]
        else:
            return items
 
    def get_counters(self) -> pd.DataFrame:
        """
        Возвращает таблицу популярности всех айтемов
        """
        if self.item_popularity is None:
            raise ValueError("ListeningTimeCounter model has not been fitted. Call .fit() first.")
        return self.item_popularity.copy()
 
    def save_counters(self, path: str):
        """
        Сохраняет счётчики популярности в parquet-файл.
        Если файл существует — добавляет колонку с именем counter_name.
 
        Args:
            path: str – Путь к parquet-файлу
        """
        counters = self.get_counters().set_index('item_id').rename(
            columns={'popularity_score': self.counter_name}
        )
 
        print(f"Saving ListeningTimeCounter counters to {path}...")
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            merged = existing.merge(counters, left_index=True, right_index=True, how='outer')
            merged.to_parquet(path)
        else:
            counters.to_parquet(path)
        print(f"ListeningTimeCounter counters saved successfully as column '{self.counter_name}'!")
 
    def save_model(self, directory: str = "."):
        """
        Сохраняет модель и все поля через joblib по пути directory/{counter_name}.joblib
 
        Args:
            directory: str – Директория для сохранения
        """
        if self.item_popularity is None:
            raise ValueError("ListeningTimeCounter model has not been fitted. Call .fit() first.")
 
        model_state = {
            'counter_name': self.counter_name,
            'target_event_type': self.target_event_type,
            'listening_time_col': self.listening_time_col,
            'item_popularity': self.item_popularity,
        }
 
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{self.counter_name}.joblib")
        print(f"Saving ListeningTimeCounter model to {model_path}...")
        joblib.dump(model_state, model_path)
        print(f"ListeningTimeCounter model saved successfully!")
 
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'ListeningTimeCounter':
        """
        Загружает модель из сохранённого состояния.
 
        Args:
            model_path: str – Путь к .joblib файлу
 
        Returns:
            ListeningTimeCounter: Восстановленный экземпляр модели
        """
        print(f"Loading ListeningTimeCounter model from {model_path}...")
        model_state = joblib.load(model_path)
 
        instance = cls(counter_name=model_state['counter_name'])
        instance.target_event_type = model_state['target_event_type']
        instance.listening_time_col = model_state['listening_time_col']
        instance.item_popularity = model_state['item_popularity']
 
        print(f"ListeningTimeCounter model loaded successfully from {model_path}!")
        return instance