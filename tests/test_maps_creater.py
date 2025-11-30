import polars as pl
from models.utils import build_id_maps
from pathlib import Path
from tqdm import tqdm

def test_build_id_maps_basic():
    train = pl.DataFrame({
        "uid": ["u1", "u2", "u1", "u3"],
        "item_id": [10, 20, 10, 30],
        "rating": [1, 2, 3, 4]
    }).lazy()

    user_map, item_map = build_id_maps(train)

    # Проверяем, что все уникальные uid присутствуют
    assert set(user_map.keys()) == {"u1", "u2", "u3"}
    # Проверяем, что индексы идут от 0 подряд
    assert sorted(user_map.values()) == [0, 1, 2]

    # Проверяем item_id
    assert set(item_map.keys()) == {10, 20, 30}
    assert sorted(item_map.values()) == [0, 1, 2]


def test_build_id_maps_no_duplicates():
    train = pl.DataFrame({
        "uid": ["a", "a", "b", "b", "c"],
        "item_id": [1, 1, 2, 2, 3],
    }).lazy()

    user_map, item_map = build_id_maps(train)

    # Проверяем, что каждое значение маппинга уникально
    assert len(user_map.values()) == len(set(user_map.values()))
    assert len(item_map.values()) == len(set(item_map.values()))

    # Проверяем, что количество индексов равно количеству уникальных id
    assert len(user_map) == 3    # a, b, c
    assert len(item_map) == 3    # 1, 2, 3

