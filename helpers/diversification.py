import polars as pl
import pandas as pd
from collections import defaultdict
from pathlib import Path


class DiversificationByArtistAlbum:
    """
    Простая диверсификация кандидатов по artist_id и album_id.

    Идея:
      - идём по списку кандидатов (в порядке убывания скоринга),
      - считаем, сколько треков уже взято у каждого артиста / альбома,
      - если лимит превышен — пропускаем трек.

    Ожидаемый формат items_meta (params.datasets.items_meta):
      - item_id
      - artist_id (или artist_is — подстрахуемся)
      - album_id
    """

    def __init__(
        self,
        items_meta_path: str | Path,
        max_per_artist: int = 3,
        max_per_album: int = 3,
    ):
        self.items_meta_path = Path(items_meta_path)
        self.max_per_artist = max_per_artist
        self.max_per_album = max_per_album

        # item_id -> (artist_id, album_id)
        self.item2artist: dict = {}
        self.item2album: dict = {}

        self._load_items_meta()

    def _load_items_meta(self):
        """
        Читаем items_meta из params.datasets.items_meta и строим словари:
        item_id -> artist_id, item_id -> album_id
        """
        # можно поменять на scan_csv/scan_parquet в зависимости от формата
        # предположим parquet; при необходимости легко сменить
        lf = pl.scan_parquet(str(self.items_meta_path))

        cols = lf.collect().columns
        artist_col = "artist_id"

        meta = (
            lf.select(["item_id", artist_col, "album_id"])
              .collect()
        )

        self.item2artist = dict(meta.select(["item_id", artist_col]).iter_rows())
        self.item2album = dict(meta.select(["item_id", "album_id"]).iter_rows())

    def _diversify_item_ids(self, item_ids: list[int | str]) -> list[int | str]:
        """
        Собственно алгоритм диверсификации по списку item_id.
        """
        artist_cnt = defaultdict(int)
        album_cnt = defaultdict(int)

        diversified: list[int | str] = []

        for item_id in item_ids:
            artist_id = self.item2artist.get(item_id, None)
            album_id = self.item2album.get(item_id, None)

            # если нет метаданных — считаем, что ай̆тем можно брать без ограничений
            artist_ok = (
                artist_id is None or artist_cnt[artist_id] < self.max_per_artist
            )
            album_ok = (
                album_id is None or album_cnt[album_id] < self.max_per_album
            )

            if artist_ok and album_ok:
                diversified.append(item_id)
                if artist_id is not None:
                    artist_cnt[artist_id] += 1
                if album_id is not None:
                    album_cnt[album_id] += 1

        return diversified

    def diversify(self, candidates):
        """
        Принимает кандидатов в одном из форматов:
          - list[item_id]
          - pandas.DataFrame с колонкой "item_id"
          - polars.DataFrame с колонкой "item_id"

        Возвращает тот же тип, но уже после диверсификации.
        """
        if isinstance(candidates, (list, tuple)):
            item_ids = list(candidates)
            diversified_ids = self._diversify_item_ids(item_ids)
            return diversified_ids

        if isinstance(candidates, pd.DataFrame):
            if "item_id" not in candidates.columns:
                raise ValueError("В pandas.DataFrame должна быть колонка 'item_id'")
            item_ids = candidates["item_id"].tolist()
            diversified_ids = self._diversify_item_ids(item_ids)
            mask = candidates["item_id"].isin(diversified_ids)
            # сохраняем порядок исходных кандидатов
            return candidates[mask]

        if isinstance(candidates, pl.DataFrame):
            if "item_id" not in candidates.columns:
                raise ValueError("В polars.DataFrame должна быть колонка 'item_id'")
            item_ids = candidates["item_id"].to_list()
            diversified_ids = set(self._diversify_item_ids(item_ids))
            return candidates.filter(pl.col("item_id").is_in(diversified_ids))

        # если формат неожиданный — просто вернём как есть
        return candidates