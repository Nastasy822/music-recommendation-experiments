import polars as pl
import pandas as pd


class CadidatesFiltration:
    def __init__(self, min_days_since_last_play: int = 2):
        """
        min_days_since_last_play — сколько дней должно пройти с последнего прослушивания,
        чтобы трек можно было снова рекомендовать.
        """
        self.min_days_since_last_play = min_days_since_last_play
        self.user_recent_items: dict = {}  # uid -> set(item_id)

    def fit(self, train_df: pl.LazyFrame | pl.DataFrame):
        """
        train_df: поларсовский датафрейм/ленивый датафрейм с колонками:
            - uid
            - item_id
            - timestamp (unix sec)
            - event_type (фильтруем по "listen")
        """
        ldf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df

        # Максимальный timestamp в датасете — считаем "текущее" время
        max_ts = ldf.select(pl.col("timestamp").max().alias("max_ts")).collect()["max_ts"][0]
        two_days_sec = self.min_days_since_last_play * 24 * 3600

        # Последнее прослушивание по каждой паре (uid, item_id)
        last_listens = (
            ldf
            .filter(pl.col("event_type") == "listen")
            .group_by(["uid", "item_id"])
            .agg([
                pl.col("timestamp").max().alias("last_listen_ts"),
            ])
            .with_columns([
                ((pl.lit(max_ts) - pl.col("last_listen_ts")) / 86400.0)
                .alias("days_since_last_play")
            ])
            .filter(pl.col("days_since_last_play") < self.min_days_since_last_play)
            .select(["uid", "item_id"])
            .collect()
        )

        # Строим dict: uid -> set(item_id), которые нужно фильтровать
        self.user_recent_items = {}
        for uid, item_id in last_listens.iter_rows():
            self.user_recent_items.setdefault(uid, set()).add(item_id)

    def filter(self, uid, candidates):
        """
        candidates:
          - либо список item_id,
          - либо pandas.DataFrame с колонкой "item_id",
          - либо polars.DataFrame с колонкой "item_id".

        Возвращает тот же тип, но с выкинутыми треками,
        которые юзер слушал менее min_days_since_last_play дней назад.
        """
        banned = self.user_recent_items.get(uid, set())
        if not banned:
            return candidates  # нечего фильтровать

        # 1) Список item_id
        if isinstance(candidates, (list, tuple)):
            return [item for item in candidates if item not in banned]

        # 2) pandas.DataFrame
        if isinstance(candidates, pd.DataFrame):
            return candidates[~candidates["item_id"].isin(banned)]

        # 3) polars.DataFrame
        if isinstance(candidates, pl.DataFrame):
            return candidates.filter(~pl.col("item_id").is_in(list(banned)))

        # fallback: ничего не делаем
        return candidates