EVENT_MAP = {
    "like": "like",
    "unlike": "dislike",
    "dislike": "dislike",
    "undislike": "like",
    "listen": "listen",
}


def clean_listen_duplicates(df):
    """
    Очистка событий прослушивания:
    - оставляем только события с event_type == "listen";
    - аггрегируем дубли по (uid, item_id, event_type, timestamp):
        * played_ratio_pct -> max
        * is_organic       -> mean
        * track_length_seconds -> first
    - удаляем строки с некорректным is_organic (оставляем только 0.0 или 1.0).
    """
    listen_mask = df["event_type"] == "listen"
    df_listen = df.loc[listen_mask].copy()

    grouped = (
        df_listen
        .groupby(["uid", "item_id", "event_type", "timestamp"], as_index=False)
        .agg({
            "played_ratio_pct": "max",
            "is_organic": "mean",
            "track_length_seconds": "first",
        })
    )

    # оставляем только корректные значения is_organic (0 или 1)
    grouped = grouped[grouped["is_organic"].isin([0.0, 1.0])]

    return grouped
    
def remove_not_valid_listen_data(df_listen):
    """
    Удаляет некорректные события прослушивания:
    played_ratio_pct должен быть в диапазоне [0, 100].
    """
    valid_mask = df_listen["played_ratio_pct"].between(0, 100, inclusive="both")
    return df_listen[valid_mask].copy()

def cut_track_len(df, min_limit: int, max_limit: int):
    """
    Фильтрует треки по длительности.
    Оставляет только те записи, у которых track_length_seconds
    находится в диапазоне [min_limit, max_limit].
    """
    mask = df["track_length_seconds"].between(min_limit, max_limit, inclusive="both")
    return df[mask].copy()


def convert_reaction(df):
    """
    Оставляет только последнюю реакцию пользователя на каждый (uid, item_id).
    Учитываются только reaction-события: like, unlike, dislike, undislike.
    Последовательность определяется порядком строк (индексом) или timestamp,
    если датафрейм предварительно отсортирован.
    """

    reaction_events = ["like", "unlike", "dislike", "undislike"]

    # Берём только реакции
    reactions = df[df["event_type"].isin(reaction_events)].copy()

    # Группируем и берём последнюю реакцию
    last_reactions = (
        reactions.groupby(["uid", "item_id"], as_index=False)
                 .last()
                 .copy()
    )

    return last_reactions
                                                                                                                                                                           
                                                                                                                                                                           
def rename_events(df):
    """
    Приводит типы событий к единому набору:
    - unlike / dislike → dislike
    - undislike → like
    - остальные события остаются без изменений.
    """
    df = df.copy()
    df["event_type"] = df["event_type"].map(EVENT_MAP).fillna(df["event_type"])
    return df


def filter_rare_items(df, min_listens: int = 5):
    """
    Удаляет слишком редкие треки:
    - оставляет только item_id, которые встречаются >= min_listens раз.
    """
    item_counts = df.groupby("item_id").size()
    good_items = item_counts[item_counts >= min_listens].index
    
    return df[df["item_id"].isin(good_items)].copy()


def filter_rare_users(df, min_listens: int = 20):
    """
    Удаляет пользователей, у которых слишком мало прослушиваний.
    Такие пользователи не дают стабильного сигнала для обучения.
    """
    user_counts = df.groupby("uid").size()
    good_users = user_counts[user_counts >= min_listens].index

    return df[df["uid"].isin(good_users)].copy()


def remove_rer_data(train_df_listen, min_item_listens = 5, min_users_listens = 20):
    """
    Комплексная очистка:
    1. Удаление редких item_id (<5 прослушиваний)
    2. Удаление пользователей с малым числом прослушиваний (<20)
    3. Повторное удаление редких item_id (после фильтрации пользователей)
    """
    df = train_df_listen.copy()

    # 1. Удаляем редкие треки
    df = filter_rare_items(df, min_listens=5)

    # 2. Удаляем редких пользователей
    df = filter_rare_users(df, min_listens=20)

    # 3. Повторно удаляем редкие треки (распределение могло измениться)
    df = filter_rare_items(df, min_listens=5)

    return df

def remove_duplicates_by_timestamps(df):
    return df[df.duplicated(["timestamp", "uid", "event_type"], keep=False) == False]




# список шагов: (имя функции, функция, kwargs)
PREPROCESSING = [
    ("clean_listen_duplicates",      clean_listen_duplicates,      {}),
    ("remove_not_valid_listen_data", remove_not_valid_listen_data, {}),
    ("cut_track_len",                cut_track_len,                {"min_limit": 60, "max_limit": 350}),
    ("remove_rer_data",              remove_rer_data,              {}),
    ("remove_duplicates_by_timestamps",              remove_duplicates_by_timestamps,              {}),
]

def run_listen_pipeline(df, list_of_fun):
    """
    Прогоняет датафрейм через пайплайн предобработки
    и собирает статистику по количеству строк на каждом шаге.
    """
    stats = []
    
    current = df.copy()
    stats.append(("raw", len(current)))  # исходное количество строк

    for step_name, func, kwargs in PREPROCESSING:
        if step_name in list_of_fun:
            current = func(current, **kwargs)
            stats.append((step_name, len(current)))

    return current, stats