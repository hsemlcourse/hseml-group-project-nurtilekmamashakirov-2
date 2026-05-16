import pandas as pd
import numpy as np
import sys



def split_data_by_timestamp(df: pd.DataFrame, train_threshold: float = 0.8):
    """
    Разделение интеракций на обучающую и тестовую выборку по timesplit

    Args:
        df: pd.DataFrame: Датафрейм, который содержит колонку timestamp
        train_threshold (float): Доля интеракций в обучающей выборке

    Returns:
        tuple: (train_df, test_df).
    """
    # Sort the DataFrame by timestamp to ensure a chronological split
    df_sorted = df.sort_values(by='timestamp').reset_index(drop=True)

    # Calculate the number of rows for the training set
    train_size = int(len(df_sorted) * train_threshold)

    # Split the DataFrame into training and testing sets
    train_df = df_sorted.iloc[:train_size]
    test_df = df_sorted.iloc[train_size:]

    print(f"Training set size: {len(train_df)} rows")
    print(f"Testing set size: {len(test_df)} rows")

    return train_df, test_df


def get_ground_truth(
    test_df: pd.DataFrame,
    played_ratio_threshold: float = 80.0,
    user_col: str = 'uid',
    item_col: str = 'item_id',
    event_type_col: str = 'event_type',
    played_ratio_col: str = 'played_ratio_pct',
) -> pd.DataFrame:
    """
    Возвращает релевантные айтемы для каждого пользователя.

    Айтем считается релевантным если выполняется хотя бы одно условие:
        - есть событие 'like' И нет 'unlike' И нет 'dislike'
        - played_ratio_pct > threshold

    Args:
        test_df: pd.DataFrame – тестовый датафрейм с событиями
        played_ratio_threshold: float – порог played_ratio_pct для релевантности (default: 50.0)
        user_col: str – колонка с id пользователя
        item_col: str – колонка с id айтема
        event_type_col: str – колонка с типом события
        played_ratio_col: str – колонка с процентом прослушивания

    Returns:
        pd.DataFrame: индекс = user_col, колонка 'relevant_items' со списком релевантных item_id
    """
    # Пары (user, item) для каждого типа события
    liked    = set(zip(*test_df[test_df[event_type_col] == 'like'][[user_col, item_col]].values.T)) \
               if (test_df[event_type_col] == 'like').any() else set()
    unliked  = set(zip(*test_df[test_df[event_type_col] == 'unlike'][[user_col, item_col]].values.T)) \
               if (test_df[event_type_col] == 'unlike').any() else set()
    disliked = set(zip(*test_df[test_df[event_type_col] == 'dislike'][[user_col, item_col]].values.T)) \
               if (test_df[event_type_col] == 'dislike').any() else set()

    # Пары (user, item) с высоким played_ratio
    high_ratio_mask = test_df[played_ratio_col] > played_ratio_threshold
    high_ratio = set(zip(*test_df[high_ratio_mask][[user_col, item_col]].values.T)) \
                 if high_ratio_mask.any() else set()

    # Релевантные по лайкам: есть like, нет unlike, нет dislike
    liked_relevant = liked - unliked - disliked

    # Объединяем оба критерия
    all_relevant = liked_relevant | high_ratio

    if not all_relevant:
        return pd.DataFrame(columns=[user_col, 'relevant_items']).set_index(user_col)

    # Группируем в словарь user -> [item, ...]
    user_items: dict[int, list] = {}
    for user, item in all_relevant:
        user_items.setdefault(user, []).append(item)

    ground_truth = pd.DataFrame(
        [(user, items) for user, items in user_items.items()],
        columns=[user_col, 'relevant_items']
    ).set_index(user_col)

    return ground_truth

# TODO отрефакторить
def add_als_dot_product_features(
    df: pd.DataFrame,
    als_decomposition_names: list[str],
    user_embeddings_path: str,
    item_embeddings_path: str
) -> pd.DataFrame:
    """
    Adds new features to the DataFrame based on the scalar product of ALS user and item embeddings.
    Optimized for memory by using df.apply with direct map lookups instead of intermediate large Series of lists.

    Args:
        df (pd.DataFrame): The input DataFrame (e.g., train_df) containing 'uid' and 'item_id'.
        als_decomposition_names (list[str]): A list of ALS decomposition names (column names
                                            in the embeddings files) to use for feature creation.
        user_embeddings_path (str): Path to the user embeddings parquet file.
        item_embeddings_path (str): Path to the item embeddings parquet file.

    Returns:
        pd.DataFrame: The DataFrame with added scalar product features.
    """
    df_with_features = df
    print(f"Размер исходного df_with_features: {sys.getsizeof(df_with_features) // 1024 // 1024}MB")

    # Вспомогательная функция для df.apply
    def _calculate_dot_product_for_row(row, user_map, item_map, factors):
        uid = row['uid']
        item_id = row['item_id']

        user_emb = user_map.get(uid)
        item_emb = item_map.get(item_id)

        if user_emb is not None and item_emb is not None:
            # Преобразуем списки в массивы numpy только для текущей строки
            return np.dot(np.array(user_emb), np.array(item_emb))
        else:
            return 0.0 # Значение по умолчанию для отсутствующих эмбеддингов

    for decomp_name in als_decomposition_names:
        print(f"Вычисление скалярного произведения для разложения: {decomp_name} с использованием df.apply...")

        # Загружаем DataFrame эмбеддингов для текущего разложения внутри цикла
        # Это уменьшает пиковое потребление памяти при обработке нескольких decomp_names
        print(f"Загрузка эмбеддингов пользователей для {decomp_name}...")
        user_embeddings_df = pd.read_parquet(user_embeddings_path)
        print(f"user_embeddings_df: {sys.getsizeof(user_embeddings_df) // 1024 // 1024}MB")
        print(f"Загрузка эмбеддингов элементов для {decomp_name}...")
        item_embeddings_df = pd.read_parquet(item_embeddings_path)
        print(f"item_embeddings_df: {sys.getsizeof(item_embeddings_df) // 1024 // 1024}MB")

        # Надежная проверка и установка имен столбцов
        if len(user_embeddings_df.columns) == 1 and user_embeddings_df.columns[0] != decomp_name:
            user_embeddings_df.rename(columns={user_embeddings_df.columns[0]: decomp_name}, inplace=True)
            print(f"Переименован столбец пользовательских эмбеддингов в '{decomp_name}'. Текущие столбцы: {user_embeddings_df.columns.tolist()}")
        elif decomp_name not in user_embeddings_df.columns:
            print(f"Ошибка: Ожидаемый столбец '{decomp_name}' не найден в пользовательских эмбеддингах. Доступные столбцы: {user_embeddings_df.columns.tolist()}")
            raise KeyError(f"Столбец '{decomp_name}' не найден в DataFrame пользовательских эмбеддингов.")

        if len(item_embeddings_df.columns) == 1 and item_embeddings_df.columns[0] != decomp_name:
            item_embeddings_df.rename(columns={item_embeddings_df.columns[0]: decomp_name}, inplace=True)
            print(f"Переименован столбец эмбеддингов элементов в '{decomp_name}'. Текущие столбцы: {item_embeddings_df.columns.tolist()}")
        elif decomp_name not in item_embeddings_df.columns:
            print(f"Ошибка: Ожидаемый столбец '{decomp_name}' не найден в эмбеддингах элементов. Доступные столбцы: {item_embeddings_df.columns.tolist()}")
            raise KeyError(f"Столбец '{decomp_name}' не найден в DataFrame эмбеддингов элементов.")

        # Создаем отображения для быстрого поиска
        user_embedding_map = user_embeddings_df[decomp_name].to_dict()
        print(f"user_embedding_map: {sys.getsizeof(user_embedding_map) // 1024 // 1024}MB")
        del user_embeddings_df # Освобождаем память раньше
        item_embedding_map = item_embeddings_df[decomp_name].to_dict()
        print(f"item_embedding_map: {sys.getsizeof(item_embedding_map) // 1024 // 1024}MB")
        del item_embeddings_df # Освобождаем память раньше

        # Определяем размерность факторов для массива 0.0 по умолчанию, если эмбеддинг отсутствует
        factors = len(next(iter(user_embedding_map.values()))) if user_embedding_map else 0
        print(f"factors: {sys.getsizeof(factors) // 1024 // 1024}MB")
        # Применяем вспомогательную функцию построчно
        df_with_features[f'{decomp_name}_dot_product'] = df_with_features.apply(
            lambda row: _calculate_dot_product_for_row(row, user_embedding_map, item_embedding_map, factors),
            axis=1
        )
        print(f"df_with_features после скалярного произведения: {sys.getsizeof(df_with_features) // 1024 // 1024}MB")

        # Явно удаляем отображения для освобождения памяти
        del user_embedding_map
        del item_embedding_map
        print(f"Отображения эмбеддингов для {decomp_name} удалены из памяти.")

    return df_with_features