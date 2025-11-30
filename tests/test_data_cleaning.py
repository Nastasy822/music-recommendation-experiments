import polars as pl
import pytest
from helpers.data_cleaning import *
import helpers.data_cleaning
import polars as pl
from pathlib import Path


def test_remove_all_duplicates():
    df = pl.DataFrame({
        "timestamp": [1, 1, 2],
        "uid": ["u1", "u1", "u2"],
        "event_type": ["click", "click", "view"],
        "value": [10, 20, 30]
    })

    result = remove_duplicates_by_timestamps(df.lazy()).collect()

    assert result.shape == (1, 4)
    assert result.row(0) == (2, "u2", "view", 30)


def test_keep_unique_rows():
    df = pl.DataFrame({
        "timestamp": [1, 2, 3],
        "uid": ["u1", "u2", "u3"],
        "event_type": ["click", "click", "view"],
        "value": [10, 20, 30]
    })

    result = remove_duplicates_by_timestamps(df.lazy()).collect()

    assert result.shape == df.shape
    assert result.to_dict(as_series=False) == df.to_dict(as_series=False)




def test_get_listen_data_only_listen_events():
    df = pl.DataFrame({
        "uid": ["u1", "u2", "u3"],
        "event_type": ["listen", "click", "listen"],
        "item_id": [1, 2, 3],
    }).lazy()

    result = get_listen_data(df).collect()

    assert result.shape == (2, 3)
    assert result["event_type"].to_list() == ["listen", "listen"]
    assert result["uid"].to_list() == ["u1", "u3"]


def test_get_not_listen_data_excludes_listen():
    df = pl.DataFrame({
        "uid": ["u1", "u2", "u3"],
        "event_type": ["listen", "click", "open"],
        "item_id": [1, 2, 3],
    }).lazy()

    result = get_not_listen_data(df).collect()

    assert result.shape == (2, 3)
    assert set(result["event_type"].to_list()) == {"click", "open"}
    assert "listen" not in result["event_type"].to_list()



def test_filter_rare_users_min_listens():
    df = pl.DataFrame({
        # u1 – 3 события, u2 – 5 событий
        "uid":      ["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2"],
        "item_id":  [10,   11,   12,   20,   21,   22,   23,   24],
        "event_type": ["listen"] * 8,
    }).lazy()

    result = filter_rare_users(df, min_listens=4).collect()

    # остаётся только u2, у которого 5 событий
    assert set(result["uid"].to_list()) == {"u2"}
    assert result.shape[0] == 5


def test_filter_rare_users_min_listens():
    df = pl.DataFrame({
        "uid": ["u1"] * 3 + ["u2"] * 5,
        "item_id": [10, 11, 12, 20, 20, 21, 22, 23,],
        "event_type": ["listen"] * 8,
    }).lazy()

    # u1 – 3 события, u2 – 5 событий
    result = filter_rare_users(df, min_listens=4).collect()

    assert set(result["uid"].to_list()) == {"u2"}
    assert result.shape[0] == 5


def test_cut_track_len_inclusive_bounds():
    df = pl.DataFrame({
        "uid": ["u1", "u2", "u3", "u4", "u5"],
        "track_length_seconds": [30, 60, 200, 350, 400],
    }).lazy()

    result = cut_track_len(df, min_limit=60, max_limit=350).collect()

    # должны остаться треки с длиной 60, 200, 350 (границы включены)
    assert result["track_length_seconds"].to_list() == [60, 200, 350]
    assert result["uid"].to_list() == ["u2", "u3", "u4"]


def test_convert_reaction_takes_only_reactions_and_last_event():
    df = pl.DataFrame({
        "uid": ["u1", "u1", "u1", "u2"],
        "item_id": [10,   10,   10,   20],
        "event_type": ["like", "dislike", "unlike", "listen"],
        "ts": [1, 2, 3, 4],
    }).lazy()

    result = convert_reaction(df).collect()

    # u1-10: реакции like -> dislike -> unlike → должен остаться только последний (unlike)
    # u2-20: только "listen" → вообще не должен попасть в результат
    assert result.shape[0] == 1
    row = result.row(0, named=True)

    assert row["uid"] == "u1"
    assert row["item_id"] == 10
    assert row["event_type"] == "unlike"
    # проверяем, что "последний" действительно по колонке ts
    assert row["ts"] == 3


def test_convert_reaction_multiple_users_and_items():
    df = pl.DataFrame({
        "uid":       ["u1", "u1", "u2", "u2", "u3"],
        "item_id":   [10,   10,   20,   20,   30],
        "event_type": ["like", "dislike", "like", "undislike", "dislike"],
        "ts":        [1,     2,     1,    2,         3],
    }).lazy()

    result = convert_reaction(df).sort(["uid", "item_id"]).collect()

    assert result.shape[0] == 3

    # u1-10: last → "dislike"
    r_u1 = result.filter((pl.col("uid") == "u1") & (pl.col("item_id") == 10)).row(0, named=True)
    assert r_u1["event_type"] == "dislike"
    assert r_u1["ts"] == 2

    # u2-20: last → "undislike"
    r_u2 = result.filter((pl.col("uid") == "u2") & (pl.col("item_id") == 20)).row(0, named=True)
    assert r_u2["event_type"] == "undislike"
    assert r_u2["ts"] == 2

    # u3-30: только один event → он и остаётся
    r_u3 = result.filter((pl.col("uid") == "u3") & (pl.col("item_id") == 30)).row(0, named=True)
    assert r_u3["event_type"] == "dislike"
    assert r_u3["ts"] == 3


def test_rename_events_basic_mapping():
    df = pl.DataFrame({
        "uid": ["u1", "u2", "u3", "u4"],
        "item_id": [10, 20, 30, 40],
        "event_type": ["like", "unlike", "dislike", "undislike"],
    }).lazy()

    result = rename_events(df).collect()

    # Проверяем по EVENT_MAP:
    # like       -> like
    # unlike     -> dislike
    # dislike    -> dislike
    # undislike  -> like
    assert result["event_type"].to_list() == [
        EVENT_MAP["like"],
        EVENT_MAP["unlike"],
        EVENT_MAP["dislike"],
        EVENT_MAP["undislike"],
    ]

    # Остальные колонки не изменились
    assert result["uid"].to_list() == ["u1", "u2", "u3", "u4"]
    assert result["item_id"].to_list() == [10, 20, 30, 40]


def test_rename_events_keeps_unknown_values():
    df = pl.DataFrame({
        "uid": ["u1", "u2"],
        "item_id": [10, 20],
        "event_type": ["listen", "some_weird_event"],
    }).lazy()

    result = rename_events(df).collect()

    # "listen" есть в EVENT_MAP и должен остаться "listen"
    assert result["event_type"][0] == EVENT_MAP["listen"]

    # "some_weird_event" нет в EVENT_MAP -> должно остаться как есть
    assert result["event_type"][1] == "some_weird_event"


def test_rename_events_does_not_change_schema_except_event_type_values():
    df = pl.DataFrame({
        "uid": ["u1"],
        "item_id": [42],
        "event_type": ["unlike"],
        "ts": [123456789],
    }).lazy()

    result = rename_events(df).collect()

    # Имена колонок те же
    assert result.columns == ["uid", "item_id", "event_type", "ts"]

    # Значения в других колонках не тронуты
    assert result["uid"].to_list() == ["u1"]
    assert result["item_id"].to_list() == [42]
    assert result["ts"].to_list() == [123456789]

    # event_type заменён по EVENT_MAP
    assert result["event_type"].to_list() == [EVENT_MAP["unlike"]]



