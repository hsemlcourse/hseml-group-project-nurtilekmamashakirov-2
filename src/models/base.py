from abc import ABC, abstractclassmethod


class BaseModel(ABC):
    """
    Базовый класс для рекоммендаций
    """
    @abstractclassmethod
    def fit(self):
        pass

    @abstractclassmethod
    def recommend_items(self):
        pass