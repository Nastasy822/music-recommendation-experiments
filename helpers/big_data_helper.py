import polars as pl
import os
import tempfile
import shutil
from pathlib import Path


def process_in_batches(lazy_df: pl.LazyFrame, sort_column: str, batch_func, output_dir: str, batch_size: int = 100_000):
    total_rows = lazy_df.collect().height
    offset = 0
    batch_idx = 0
    
    while offset < total_rows:
        batch_df = lazy_df.slice(offset, batch_size).collect()
        processed = batch_func(batch_df)
        filename = os.path.join(output_dir, f"batch_{batch_idx}.parquet")
        processed.write_parquet(filename)
        offset += batch_size
        batch_idx += 1

        
def merge_parquet_files(input_dir: str, output_file: str):
    parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet")]
    
    lazy_frames = [pl.scan_parquet(f) for f in parquet_files]
    
    combined_lazy = pl.concat(lazy_frames)
    
    combined_lazy.collect().write_parquet(output_file)
    

def apply_function_by_batch(input_path, output_path , fun, column_filtration, batch_size = 10_000_000):
    data = pl.scan_parquet(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        output_dir = tmp_path / "tmp"
        output_dir.mkdir(parents=True, exist_ok=True)

        process_in_batches(
            data,
            column_filtration,
            fun,
            str(output_dir),
            batch_size=10_000_000,
        )

        merged_path = tmp_path / "tmp.parquet"
        merge_parquet_files(str(output_dir), str(merged_path))

        shutil.move(str(merged_path), output_path)



def estimate_parquet_ram_usage(path: str, sample_rows: int = 10_000):
    print(f"\n📁 Файл: {path}")
    print(f"📊 Сэмпл строк: {sample_rows:,}")

    df_sample = pl.read_parquet(path, n_rows=sample_rows)
    size_sample = df_sample.estimated_size()
    print(f"🔹 Размер sample: {size_sample/1024**2:.2f} MB")

    lf = pl.scan_parquet(path)
    total_rows = lf.select(pl.len()).collect().item()
    print(f"🔹 Всего строк в файле: {total_rows:,}")

    estimated_total_bytes = size_sample * (total_rows / sample_rows)
    estimated_total_gb = estimated_total_bytes / 1024**3

    print(f"\n📐 Примерная оценка объёма данных в памяти: {estimated_total_gb:.2f} GB")



def concat_files(lf1_path, lf2_path, result_path):
    lf1 = pl.scan_parquet(lf1_path)
    lf2 = pl.scan_parquet(lf2_path)
    
    lf1_cols = lf1.collect_schema().names()
    lf2_cols = lf2.collect_schema().names()

    lf2_aligned = lf2.select(
        [pl.col(c) if c in lf2_cols else pl.lit(None).alias(c) for c in lf1_cols]
    )

    lf = pl.concat([lf1, lf2_aligned])
    
    lf.sink_parquet(result_path)
