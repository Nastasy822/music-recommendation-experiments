import logging
import polars as pl

from stages.base_stage import BaseStage
from helpers.big_data_helper import (
    estimate_parquet_ram_usage,
    apply_function_by_batch,
    concat_files,
)
from helpers.params_provider import ParamsProvider
from helpers.train_test_split import time_split_with_gap
from helpers.data_cleaning import (
    get_listen_data,
    get_not_listen_data,
    remove_duplicates_by_timestamps,
    filter_rare_items,
    filter_rare_users,
    cut_track_len,
    convert_reaction,
    rename_events,
    select_listened_data,
)

logger = logging.getLogger(__name__)


class DataPreprocessing(BaseStage):
    def __init__(self):
        super().__init__()

        self.params = ParamsProvider().get_params()

        preprocessing_params = self.params.preprocessing
        self.gap_size = preprocessing_params.gap_size
        self.test_size = preprocessing_params.test_size
        self.batch_size = preprocessing_params.batch_size

        self.source_dataset_path = self.params.datasets.source_dataset_path

    # ===== Вспомогательные методы ==============================================

    def _apply_by_batch(self, src_path: str, dst_path: str, func, column: str) -> None:
        """Обёртка над apply_function_by_batch с использованием batch_size из параметров."""
        logger.debug(
            "Applying %s by batch on %s -> %s (column=%s, batch_size=%s)",
            getattr(func, "__name__", str(func)),
            src_path,
            dst_path,
            column,
            self.batch_size,
        )
        apply_function_by_batch(
            src_path,
            dst_path,
            func,
            column,
            batch_size=self.batch_size,
        )

    def _split_train_test(self) -> None:
        """Сканирование исходного датасета, split на train/test и сохранение."""
        logger.info("Step 1: scanning source parquet: %s", self.source_dataset_path)
        data_lf = pl.scan_parquet(self.source_dataset_path)

        estimate_parquet_ram_usage(self.source_dataset_path)

        logger.info(
            "Step 1: splitting to train/test (test_size=%s, gap_size=%s)",
            self.test_size,
            self.gap_size,
        )
        train_lf, test_lf = time_split_with_gap(
            data_lf,
            self.test_size,
            gap_size=self.gap_size,
        )

        logger.info("Step 1: saving train/test parquet files")
        train_lf.sink_parquet(self.params.datasets.train.source)
        test_lf.sink_parquet(self.params.datasets.test.source)

        estimate_parquet_ram_usage(self.params.datasets.train.source)
        estimate_parquet_ram_usage(self.params.datasets.test.source)

    def _build_listen_and_likes(self) -> None:
        """Выделение listen и likes для train и test из source."""
        train_paths = self.params.datasets.train
        test_paths = self.params.datasets.test

        logger.info("Step 2: building listen datasets from source")
        # TRAIN listen
        self._apply_by_batch(
            train_paths.source,
            train_paths.listen,
            get_listen_data,
            "timestamp",
        )
        # TEST listen
        self._apply_by_batch(
            test_paths.source,
            test_paths.listen,
            get_listen_data,
            "timestamp",
        )

        logger.info("Step 2: building likes datasets from source")
        # TRAIN likes
        self._apply_by_batch(
            train_paths.source,
            train_paths.likes,
            get_not_listen_data,
            "timestamp",
        )
        # TEST likes
        self._apply_by_batch(
            test_paths.source,
            test_paths.likes,
            get_not_listen_data,
            "timestamp",
        )

    def _clean_train_listen(self) -> None:
        """Очистка train.listen: дубликаты, редкие айтемы/юзеры, длина трека."""
        listen_path = self.params.datasets.train.listen

        logger.info("Step 3: cleaning train.listen (remove duplicates)")
        self._apply_by_batch(
            listen_path,
            listen_path,
            remove_duplicates_by_timestamps,
            "timestamp",
        )

        logger.info("Step 3: cleaning train.listen (filter rare items)")
        self._apply_by_batch(
            listen_path,
            listen_path,
            filter_rare_items,
            "item_id",
        )

        logger.info("Step 3: cleaning train.listen (filter rare users)")
        self._apply_by_batch(
            listen_path,
            listen_path,
            filter_rare_users,
            "uid",
        )

        logger.info("Step 3: cleaning train.listen (cut track length)")
        self._apply_by_batch(
            listen_path,
            listen_path,
            cut_track_len,
            "timestamp",
        )

    def _process_likes(self) -> None:
        """Обработка лайков (train и test): convert_reaction + rename_events."""
        train_paths = self.params.datasets.train
        test_paths = self.params.datasets.test

        logger.info("Step 4: processing TRAIN likes (convert_reaction, rename_events)")
        self._apply_by_batch(
            train_paths.likes,
            train_paths.likes,
            convert_reaction,
            "uid",
        )
        self._apply_by_batch(
            train_paths.likes,
            train_paths.likes,
            rename_events,
            "timestamp",
        )

        logger.info("Step 4: processing TEST likes (convert_reaction, rename_events)")
        self._apply_by_batch(
            test_paths.likes,
            test_paths.likes,
            convert_reaction,
            "uid",
        )
        self._apply_by_batch(
            test_paths.likes,
            test_paths.likes,
            rename_events,
            "timestamp",
        )

    def _filter_test_listened(self) -> None:
        """
        Фильтрация тестовых файлов likes и listen через select_listened_data.
        (поведение такое же, как в исходном коде — через apply_function_by_batch).
        """
        test_paths = self.params.datasets.test

        logger.info("Step 5: filtering TEST likes with select_listened_data")
        self._apply_by_batch(
            test_paths.likes,
            test_paths.likes,
            select_listened_data,
            "timestamp",
        )

        logger.info("Step 5: filtering TEST listen with select_listened_data")
        self._apply_by_batch(
            test_paths.listen,
            test_paths.listen,
            select_listened_data,
            "timestamp",
        )

    def _concat_and_estimate(self) -> None:
        """Конкатенация listen/likes в preprocessed и оценка ram usage."""
        train_paths = self.params.datasets.train
        test_paths = self.params.datasets.test

        logger.info("Step 6: concatenating TEST listen+likes into preprocessed")
        concat_files(
            test_paths.listen,
            test_paths.likes,
            test_paths.preprocessed,
        )

        logger.info("Step 6: concatenating TRAIN listen+likes into preprocessed")
        concat_files(
            train_paths.listen,
            train_paths.likes,
            train_paths.preprocessed,
        )

        logger.info("Step 6: estimating RAM usage for preprocessed datasets")
        estimate_parquet_ram_usage(train_paths.preprocessed)
        estimate_parquet_ram_usage(test_paths.preprocessed)

    # ===== Основной пайплайн ====================================================

    def run(self) -> None:
        logger.info("=== Starting data preprocessing pipeline ===")

        self._split_train_test()
        logger.info("Step 1 completed: train/test split done")

        self._build_listen_and_likes()
        logger.info("Step 2 completed: listen/likes built for train & test")

        self._clean_train_listen()
        logger.info("Step 3 completed: train.listen cleaned")

        self._process_likes()
        logger.info("Step 4 completed: likes processed for train & test")

        self._filter_test_listened()
        logger.info("Step 5 completed: test.listen & test.likes filtered with select_listened_data")

        self._concat_and_estimate()
        logger.info("Step 6 completed: preprocessed train/test saved")

        logger.info("=== Data preprocessing pipeline finished successfully ===")