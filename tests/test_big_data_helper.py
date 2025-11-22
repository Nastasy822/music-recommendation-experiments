
from utils.big_data_helper import *
import polars as pl
from pathlib import Path

def test_concat_files_same_schema(tmp_path: Path):
    # Подготовка временных файлов
    df1 = pl.DataFrame({
        "uid": ["u1", "u2"],
        "item_id": [10, 20],
        "timestamp": [1, 2],
    })
    df2 = pl.DataFrame({
        "uid": ["u3", "u4"],
        "item_id": [30, 40],
        "timestamp": [3, 4],
    })

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"
    result = tmp_path / "result.parquet"

    df1.write_parquet(file1)
    df2.write_parquet(file2)

    # Запускаем функцию
    concat_files(str(file1), str(file2), str(result))

    # Читаем результат
    res_lf = pl.scan_parquet(str(result))
    res = res_lf.collect()

    # Проверяем форму и содержимое
    assert res.shape == (4, 3)
    assert res.columns == ["uid", "item_id", "timestamp"]

    assert res["uid"].to_list() == ["u1", "u2", "u3", "u4"]
    assert res["item_id"].to_list() == [10, 20, 30, 40]
    assert res["timestamp"].to_list() == [1, 2, 3, 4]


def test_concat_files_aligns_missing_and_extra_columns(tmp_path: Path):
    # Первый файл: базовая схема
    df1 = pl.DataFrame({
        "uid": ["u1", "u2"],
        "item_id": [10, 20],
        "timestamp": [1, 2],
    })

    # Второй файл:
    # - нет колонки "timestamp"
    # - есть лишняя колонка "extra"
    df2 = pl.DataFrame({
        "uid": ["u3", "u4"],
        "item_id": [30, 40],
        "extra": ["x", "y"],
    })

    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"
    result = tmp_path / "result.parquet"

    df1.write_parquet(file1)
    df2.write_parquet(file2)

    # Запускаем функцию
    concat_files(str(file1), str(file2), str(result))

    # Читаем результат
    res = pl.scan_parquet(str(result)).collect()

    # Схема должна совпадать со схемой первого файла
    assert res.columns == ["uid", "item_id", "timestamp"]

    # Первые две строки — из df1, вторые две — из df2
    assert res.shape == (4, 3)

    # uid и item_id просто склеиваются
    assert res["uid"].to_list() == ["u1", "u2", "u3", "u4"]
    assert res["item_id"].to_list() == [10, 20, 30, 40]

    # timestamp для строк из df2 должен быть None
    assert res["timestamp"].to_list() == [1, 2, None, None]