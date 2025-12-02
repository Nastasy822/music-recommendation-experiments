import polars as pl
from helpers.params_provider import ParamsProvider

import numpy as np  # можно удалить, если реально нигде не используется

HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS


def _as_lazy(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Гарантированно приводим вход к LazyFrame."""
    return df.lazy() if isinstance(df, pl.DataFrame) else df


# ======================================================================
#   ITEM–USER PROFILE
# ======================================================================

def build_item_user_profile(
    train_df: pl.LazyFrame | pl.DataFrame,
    items_meta: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """
    Строит профиль (uid, item_id) с имплицитными сигналами и аггрегатами по артисту/альбому.

    Ожидает в train_df колонки как минимум:
        - uid
        - item_id
        - event_type ∈ {"listen", "like", "dislike"}
        - played_ratio_pct
        - timestamp (unix time в секундах)

    В items_meta как минимум:
        - item_id
        - artist_id
        - album_id
    """

    ldf = _as_lazy(train_df)
    items_meta_lf = _as_lazy(items_meta)

    # максимальный timestamp в датасете
    max_ts = ldf.select(pl.col("timestamp").max()).collect().item()

    # 1) Имплицитный сигнал: сколько раз слушали, макс % прослушивания, последний timestamp
    train_df_implicit = (
        ldf
        .filter(pl.col("event_type") == "listen")
        .group_by(["uid", "item_id"])
        .agg([
            pl.col("timestamp").count().alias("listen_count"),
            pl.col("played_ratio_pct").max().alias("played_ratio_max"),
            pl.col("timestamp").max().alias("last_listen_ts"),
        ])
    )

    # 2) Дизлайки
    train_df_dislike = (
        ldf
        .filter(pl.col("event_type") == "dislike")
        .select(["uid", "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("dislike_flag"))
    )

    # 3) Лайки
    train_df_like = (
        ldf
        .filter(pl.col("event_type") == "like")
        .select(["uid", "item_id"])
        .unique()
        .with_columns(pl.lit(1).alias("like_flag"))
    )

    # 4) Full join по uid, item_id
    merged = (
        train_df_implicit
        .join(train_df_dislike, on=["uid", "item_id"], how="full", coalesce=True)
        .join(train_df_like,    on=["uid", "item_id"], how="full", coalesce=True)
        .with_columns([
            pl.col("dislike_flag").fill_null(0),
            pl.col("like_flag").fill_null(0),
        ])
    )

    # 5) Добавляем "дней с последнего прослушивания"
    result = (
        merged
        .with_columns([
            ((pl.lit(max_ts) - pl.col("last_listen_ts")) / DAY_SECONDS)
            .alias("ui_days_since_last_play")
        ])
        # если у пары user–item не было прослушиваний (только лайки/дизлайки), last_listen_ts = null
        .with_columns([
            pl.col("ui_days_since_last_play").fill_null(9999)
        ])
    )

    # Присоединяем artist_id / album_id один раз
    result = result.join(
        items_meta_lf.select(["item_id", "artist_id", "album_id"]),
        on="item_id",
        how="left",
    )

    # Для artist_listen_count
    artist_counts = (
        result
        .group_by(["uid", "artist_id"])
        .agg(pl.col("listen_count").sum().alias("artist_listen_count"))
    )

    # Для album_listen_count
    album_counts = (
        result
        .group_by(["uid", "album_id"])
        .agg(pl.col("listen_count").sum().alias("album_listen_count"))
    )

    # Объединяем обратно к основному датафрейму
    result = result.join(artist_counts, on=["uid", "artist_id"], how="left")
    result = result.join(album_counts, on=["uid", "album_id"], how="left")

    return result


# ======================================================================
#   ITEM TIME PROFILE
# ======================================================================

def build_item_time_profile(
    df: pl.LazyFrame | pl.DataFrame,
    item_col: str = "item_id",
    ts_col: str = "timestamp",
) -> pl.LazyFrame:
    """
    Строит временной профиль трека (item’а).

    На выходе для каждого трека:
    - item_plays_for_time_profile  — количество логов, участвовавших в расчёте
    - item_morning_share          — доля прослушиваний утром      (06–11)
    - item_day_share              — доля прослушиваний днём       (12–17)
    - item_evening_share          — доля прослушиваний вечером    (18–23)
    - item_night_share            — доля прослушиваний ночью      (00–05)
    - item_weekday_share          — доля прослушиваний в будни    (пн–пт)
    - item_weekend_share          — доля прослушиваний в выходные (сб–вс)
    - item_avg_listen_hour        — средний час прослушивания (0–23)
    - item_avg_listen_weekday     — средний день недели (0=понедельник, ..., 6=воскресенье)

    Ожидает как минимум колонки:
        - item_col    (по умолчанию "item_id")
        - ts_col      (по умолчанию "timestamp") — unix time в секундах
    """

    ldf = _as_lazy(df)

    # Добавляем datetime, час и день недели
    ldf_with_time = (
        ldf.with_columns(
            pl.from_epoch(pl.col(ts_col), time_unit="s").alias("dt")
        )
        .with_columns([
            pl.col("dt").dt.hour().alias("hour"),
            pl.col("dt").dt.weekday().alias("weekday"),  # 0=понедельник, 6=воскресенье
        ])
    )

    # Флаги временных интервалов и будни/выходные
    ldf_flags = (
        ldf_with_time.with_columns([
            # Утро: 06–11
            pl.when(pl.col("hour").is_between(6, 11))
              .then(1).otherwise(0)
              .alias("is_morning"),

            # День: 12–17
            pl.when(pl.col("hour").is_between(12, 17))
              .then(1).otherwise(0)
              .alias("is_day"),

            # Вечер: 18–23
            pl.when(pl.col("hour").is_between(18, 23))
              .then(1).otherwise(0)
              .alias("is_evening"),

            # Ночь: 00–05
            pl.when(pl.col("hour").is_between(0, 5))
              .then(1).otherwise(0)
              .alias("is_night"),

            # Будни (0–4) и выходные (5–6)
            pl.when(pl.col("weekday") < 5)
              .then(1).otherwise(0)
              .alias("is_weekday"),

            pl.when(pl.col("weekday") >= 5)
              .then(1).otherwise(0)
              .alias("is_weekend"),
        ])
    )

    # Агрегация по item’у
    item_time_profile = (
        ldf_flags
        .group_by(item_col)
        .agg([
            pl.len().alias("item_plays_for_time_profile"),

            pl.mean("is_morning").alias("item_morning_share"),
            pl.mean("is_day").alias("item_day_share"),
            pl.mean("is_evening").alias("item_evening_share"),
            pl.mean("is_night").alias("item_night_share"),

            pl.mean("is_weekday").alias("item_weekday_share"),
            pl.mean("is_weekend").alias("item_weekend_share"),

            pl.col("hour").mean().alias("item_avg_listen_hour"),
            pl.col("weekday").mean().alias("item_avg_listen_weekday"),
        ])
    )

    return item_time_profile


# ======================================================================
#   USER TIME PROFILE
# ======================================================================

def build_user_time_profile(
    df: pl.LazyFrame | pl.DataFrame,
    uid_col: str = "uid",
    ts_col: str = "timestamp",
) -> pl.LazyFrame:
    """
    Строит временной профиль пользователя.

    На выходе для каждого пользователя:
    - user_plays_for_time_profile  — количество логов, участвовавших в расчёте
    - morning_share                — доля прослушиваний утром      (06–11)
    - day_share                    — доля прослушиваний днём       (12–17)
    - evening_share                — доля прослушиваний вечером    (18–23)
    - night_share                  — доля прослушиваний ночью      (00–05)
    - weekday_share                — доля прослушиваний в будни    (пн–пт)
    - weekend_share                — доля прослушиваний в выходные (сб–вс)
    - avg_listen_hour              — средний час прослушивания (0–23)
    - avg_listen_weekday           — средний день недели (0=понедельник, ..., 6=воскресенье)

    Ожидает как минимум колонки:
        - uid_col      (по умолчанию "uid")
        - ts_col       (по умолчанию "timestamp") — unix time в секундах
    """

    ldf = _as_lazy(df)

    ldf_with_time = (
        ldf.with_columns(
            pl.from_epoch(pl.col(ts_col), time_unit="s").alias("dt")
        )
        .with_columns([
            pl.col("dt").dt.hour().alias("hour"),
            pl.col("dt").dt.weekday().alias("weekday"),  # 0=понедельник, 6=воскресенье
        ])
    )

    # Режимы времени суток + будни/выходные
    ldf_flags = (
        ldf_with_time.with_columns([
            pl.when(pl.col("hour").is_between(6, 11))
              .then(1).otherwise(0)
              .alias("is_morning"),

            pl.when(pl.col("hour").is_between(12, 17))
              .then(1).otherwise(0)
              .alias("is_day"),

            pl.when(pl.col("hour").is_between(18, 23))
              .then(1).otherwise(0)
              .alias("is_evening"),

            pl.when(pl.col("hour").is_between(0, 5))
              .then(1).otherwise(0)
              .alias("is_night"),

            pl.when(pl.col("weekday") < 5)
              .then(1).otherwise(0)
              .alias("is_weekday"),

            pl.when(pl.col("weekday") >= 5)
              .then(1).otherwise(0)
              .alias("is_weekend"),
        ])
    )

    # Агрегация по пользователю
    user_time_profile = (
        ldf_flags
        .group_by(uid_col)
        .agg([
            pl.len().alias("user_plays_for_time_profile"),

            pl.mean("is_morning").alias("morning_share"),
            pl.mean("is_day").alias("day_share"),
            pl.mean("is_evening").alias("evening_share"),
            pl.mean("is_night").alias("night_share"),
            pl.mean("is_weekday").alias("weekday_share"),
            pl.mean("is_weekend").alias("weekend_share"),

            pl.col("hour").mean().alias("avg_listen_hour"),
            pl.col("weekday").mean().alias("avg_listen_weekday"),
        ])
    )

    return user_time_profile


# ======================================================================
#   USER MUSIC STATS
# ======================================================================

def user_music_stats(
    df: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """
    Рассчитывает для каждого пользователя:
    - user_total_plays        — общее количество прослушиваний
    - user_active_days        — количество активных дней
    - user_unique_tracks      — количество уникальных треков
    - median_daily_plays      — медианное количество прослушиваний в день
    - unique_tracks_share     — доля уникальных треков от всех прослушиваний
    - days_since_last_play    — дни с момента последнего прослушивания
    + user_lifetime_days, user_lifetime_days_safe
    + временной профиль пользователя (join build_user_time_profile)
    """

    ldf = _as_lazy(df)

    # Нормализуем дату
    ldf = ldf.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").dt.date().alias("date")
    )

    # Последний таймстамп из всего датасета
    max_ts = (
        ldf.select(pl.col("timestamp").max().alias("max_ts"))
        .collect()
        .item()
    )

    # Считаем количество прослушиваний в каждый день по пользователям
    plays_per_day = (
        ldf.group_by(["uid", "date"])
           .agg(pl.len().alias("plays_per_day"))
    )

    # Среднее (медианное) количество прослушиваний в день на пользователя
    median_daily_plays = (
        plays_per_day.group_by("uid")
                     .agg(pl.col("plays_per_day").median().alias("median_daily_plays"))
    )

    # Статистика пользователя по всей истории
    user_stats = (
        ldf.group_by("uid")
           .agg([
               pl.len().alias("user_total_plays"),
               pl.col("date").n_unique().alias("user_active_days"),
               pl.col("item_id").n_unique().alias("user_unique_tracks"),
               pl.col("timestamp").max().alias("last_timestamp"),
           ])
    )

    # Жизненный цикл пользователя (lifetime)
    user_lifetime = (
        ldf.group_by("uid")
           .agg([
               pl.col("timestamp").min().alias("first_ts"),
               pl.col("timestamp").max().alias("last_ts"),
           ])
           .with_columns([
               ((pl.lit(max_ts) - pl.col("first_ts")) / DAY_SECONDS)
               .alias("user_lifetime_days")
           ])
           .with_columns([
               pl.when(pl.col("user_lifetime_days") == 0)
                 .then(1)
                 .otherwise(pl.col("user_lifetime_days"))
                 .alias("user_lifetime_days_safe")
           ])
    )

    # Основные фичи по пользователю
    result = (
        user_stats.join(median_daily_plays, on="uid")
                  .with_columns([
                      (pl.col("user_unique_tracks") / pl.col("user_total_plays"))
                      .alias("unique_tracks_share"),
                      ((pl.lit(max_ts) - pl.col("last_timestamp")) / DAY_SECONDS)
                      .alias("days_since_last_play"),
                  ])
                  .select([
                      "uid",
                      "user_total_plays",
                      "user_active_days",
                      "user_unique_tracks",
                      "median_daily_plays",
                      "unique_tracks_share",
                      "days_since_last_play",
                  ])
    )

    result = result.join(user_lifetime, on="uid")

    # Добавляем временной профиль
    result = result.join(build_user_time_profile(df), on="uid")

    return result


# ======================================================================
#   ITEM STATS
# ======================================================================

def build_item_stats(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """
    Считает фичи популярности треков:
        item_total_plays
        item_unique_users
        item_plays_last_5d      — популярность за последние 5 дней
        item_plays_last_30d     — популярность за последние 30 дней
        item_trend              — item_plays_last_5d / item_plays_last_30d
        item_recent_popularity  — = item_plays_last_5d
        track_length_seconds
        item_age_days

    df должен содержать колонки:
        ['uid', 'item_id', 'timestamp', 'track_length_seconds']
    """

    ldf = _as_lazy(df)

    max_ts = (
        ldf.select(pl.col("timestamp").max().alias("max_ts"))
        .collect()
        .item()
    )

    FIVE_DAYS_SECONDS = 5 * DAY_SECONDS
    THIRTY_DAYS_SECONDS = 30 * DAY_SECONDS

    item_stats = (
        ldf.with_columns([
            (max_ts - pl.col("timestamp")).alias("age_sec")
        ])
        .group_by("item_id")
        .agg([
            # базовые фичи
            pl.len().alias("item_total_plays"),
            pl.col("uid").n_unique().alias("item_unique_users"),
            pl.col("track_length_seconds").max().alias("track_length_seconds"),

            # первый timestamp
            pl.col("timestamp").min().alias("first_ts"),

            # популярность за 5/30 дней
            pl.when(pl.col("age_sec") <= FIVE_DAYS_SECONDS)
              .then(1).otherwise(0)
              .sum()
              .alias("item_plays_last_5d"),

            pl.when(pl.col("age_sec") <= THIRTY_DAYS_SECONDS)
              .then(1).otherwise(0)
              .sum()
              .alias("item_plays_last_30d"),
        ])
        .with_columns([
            # возраст трека в днях
            ((max_ts - pl.col("first_ts")) / DAY_SECONDS).alias("item_age_days"),

            # тренд (с защитой от деления на 0)
            (
                pl.col("item_plays_last_5d")
                / pl.col("item_plays_last_30d").clip(lower_bound=1)
            ).alias("item_trend"),

            # недавняя популярность = plays_5d
            pl.col("item_plays_last_5d").alias("item_recent_popularity"),
        ])
    )

    item_stats = item_stats.join(build_item_time_profile(df), on="item_id")

    return item_stats
