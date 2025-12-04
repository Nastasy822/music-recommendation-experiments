import numpy as np
import pandas as pd
import polars as pl
from typing import Sequence, Any


class EGreedyTopK:
    """
    e-greedy постобработка:
      - Берём топ-K кандидатов (по score / по порядку в списке)
      - Внутри этого K доля exploration_rate заменяется объектами из "низа" списка.

    exploration_rate = N% (например, 0.2 = 20%)
    """

    def __init__(
        self,
        k: int = 10,
        exploration_rate: float = 0.2,
        score_col: str = "score",
        random_state: int | None = None,
    ):
        assert 0.0 <= exploration_rate <= 1.0
        self.k = k
        self.exploration_rate = exploration_rate
        self.score_col = score_col
        self.rng = np.random.default_rng(random_state)

    # ======== Вспомогательная логика для list ========
    def _apply_list(self, candidates: Sequence[Any]) -> list[Any]:
        """
        Для списка считаем, что он уже отсортирован по score по убыванию.
        Верх списка — самые высокие score, низ — самые низкие.
        """

        candidates = list(candidates)
        if not candidates:
            return []

        k = min(self.k, len(candidates))
        n_explore = int(round(k * self.exploration_rate))
        n_explore = min(max(n_explore, 0), k)
        n_exploit = k - n_explore

        # exploitation: топ-n_exploit
        exploit_part = candidates[:n_exploit]

        # exploration-пул: всё, что после топ-n_exploit
        tail = candidates[n_exploit:]
        if not tail or n_explore == 0:
            return candidates[:k]

        # выбираем n_explore случайных из хвоста
        if len(tail) <= n_explore:
            explore_part = tail
        else:
            idx = self.rng.choice(len(tail), size=n_explore, replace=False)
            explore_part = [tail[i] for i in idx]

        return exploit_part + explore_part

    # ======== Вспомогательная логика для pandas ========
    def _apply_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # сортируем по score по убыванию (если не отсортирован заранее)
        if self.score_col not in df.columns:
            raise ValueError(f"В DataFrame нет колонки '{self.score_col}' со score")

        df_sorted = df.sort_values(self.score_col, ascending=False)
        k = min(self.k, len(df_sorted))
        n_explore = int(round(k * self.exploration_rate))
        n_explore = min(max(n_explore, 0), k)
        n_exploit = k - n_explore

        exploit_part = df_sorted.iloc[:n_exploit]

        tail = df_sorted.iloc[n_exploit:]
        if tail.empty or n_explore == 0:
            return df_sorted.iloc[:k]

        if len(tail) <= n_explore:
            explore_part = tail
        else:
            idx = self.rng.choice(len(tail), size=n_explore, replace=False)
            explore_part = tail.iloc[idx]

        result = pd.concat([exploit_part, explore_part], axis=0)
        # можно перемешать, чтобы exploration-позиции не всегда шли в конце
        # result = result.sample(frac=1.0, random_state=self.rng.integers(1e9))
        return result

    # ======== Вспомогательная логика для polars ========
    def _apply_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        if self.score_col not in df.columns:
            raise ValueError(f"В Polars DataFrame нет колонки '{self.score_col}' со score")

        df_sorted = df.sort(by=self.score_col, descending=True)
        k = min(self.k, df_sorted.height)
        n_explore = int(round(k * self.exploration_rate))
        n_explore = min(max(n_explore, 0), k)
        n_exploit = k - n_explore

        exploit_part = df_sorted.slice(0, n_exploit)
        tail = df_sorted.slice(n_exploit)

        if tail.is_empty() or n_explore == 0:
            return df_sorted.slice(0, k)

        if tail.height <= n_explore:
            explore_part = tail
        else:
            idx = self.rng.choice(tail.height, size=n_explore, replace=False)
            explore_part = tail.take(pl.Series("idx", idx))

        result = pl.concat([exploit_part, explore_part])

        # если хочешь перемешать финальные позиции:
        # result = result.sample(frac=1.0, with_replacement=False)
        return result

    # ======== Публичный метод ========
    def apply(self, candidates):
        """
        candidates:
          - list / tuple (считаем уже отсортированным)
          - pandas.DataFrame c колонкой score_col
          - polars.DataFrame c колонкой score_col
        """
        if isinstance(candidates, (list, tuple)):
            return self._apply_list(candidates)
        if isinstance(candidates, pd.DataFrame):
            return self._apply_pandas(candidates)
        if isinstance(candidates, pl.DataFrame):
            return self._apply_polars(candidates)

        # неизвестный тип — возвращаем как есть
        return candidates
