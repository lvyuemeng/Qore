from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import types as pa_types

from qore_data import DataSettings
from qore_data.store.schema import DATASETS, Dataset

StoreBackend = Literal["auto", "parquet", "duckdb"]
StoreFilterValue = str | int | float | bool | date | datetime

_TYPE_PREDICATES = (
    (pa_types.is_date32, pl.Date, "DATE"),
    (pa_types.is_string, pl.String, "VARCHAR"),
    (pa_types.is_float64, pl.Float64, "DOUBLE"),
    (pa_types.is_int64, pl.Int64, "BIGINT"),
    (pa_types.is_boolean, pl.Boolean, "BOOLEAN"),
)


def _polars_schema(dataset: Dataset) -> dict[str, pl.DataType]:
    mapping: dict[str, pl.DataType] = {}
    for f in dataset.schema:
        mapping[f.name] = _polars_type(f.type)
    return mapping


def _polars_type(data_type: pa.DataType) -> pl.DataType:
    for predicate, polars_type, _ in _TYPE_PREDICATES:
        if predicate(data_type):
            return polars_type()
    msg = f"Unsupported type in schema mapping: {data_type}"
    raise TypeError(msg)


def _duckdb_type(data_type: pa.DataType) -> str:
    for predicate, _, duckdb_type in _TYPE_PREDICATES:
        if predicate(data_type):
            return duckdb_type
    msg = f"Unsupported DuckDB type mapping: {data_type}"
    raise TypeError(msg)


class QoreStore:
    def __init__(self, db_path: str, parquet_root: str) -> None:
        self._db_path = Path(db_path)
        self._parquet_root = Path(parquet_root)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._parquet_root.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self.register_all_views()

    @classmethod
    def from_settings(cls, settings: DataSettings) -> QoreStore:
        return cls(db_path=settings.db_path, parquet_root=settings.parquet_root)

    def _dataset(self, dataset: str) -> Dataset:
        if dataset not in DATASETS:
            msg = f"Unknown dataset: {dataset}"
            raise KeyError(msg)
        return DATASETS[dataset]

    def _dataset_root(self, dataset: str) -> Path:
        return self._parquet_root / dataset

    def register_all_views(self) -> None:
        for dataset_name in DATASETS:
            root = self._dataset_root(dataset_name)
            if not root.exists():
                self._register_empty_view(dataset_name)
                continue
            glob_path = (root / "**/*.parquet").as_posix()
            self._conn.execute(
                f"CREATE OR REPLACE VIEW {dataset_name} AS SELECT * FROM read_parquet('{glob_path}', union_by_name = true, hive_partitioning = false)"
            )

    def read(
        self,
        dataset: str,
        filters: Mapping[str, StoreFilterValue] | None = None,
        columns: list[str] | None = None,
        backend: str = "auto",
    ) -> pl.LazyFrame:
        dataset_info = self._dataset(dataset)
        schema_names = set(_polars_schema(dataset_info))
        if columns is not None:
            missing_columns = [
                column for column in columns if column not in schema_names
            ]
            if missing_columns:
                msg = f"Unknown columns for {dataset}: {missing_columns}"
                raise KeyError(msg)
        if filters:
            unknown_filters = [key for key in filters if key not in schema_names]
            if unknown_filters:
                msg = f"Unknown filter column for {dataset}: {unknown_filters[0]}"
                raise KeyError(msg)
        resolved_backend = "parquet" if backend == "auto" else backend
        if resolved_backend == "duckdb":
            return self._read_duckdb(dataset, dataset_info, filters, columns)
        return self._read_parquet(dataset, dataset_info, filters, columns)

    def read_parquet(
        self,
        dataset: str,
        filters: Mapping[str, StoreFilterValue] | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        return self.read(dataset, filters=filters, columns=columns, backend="parquet")

    def read_duckdb(
        self,
        dataset: str,
        filters: Mapping[str, StoreFilterValue] | None = None,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        return self.read(dataset, filters=filters, columns=columns, backend="duckdb")

    def _read_parquet(
        self,
        dataset: str,
        dataset_info: Dataset,
        filters: Mapping[str, StoreFilterValue] | None,
        columns: list[str] | None,
    ) -> pl.LazyFrame:
        root = self._dataset_root(dataset)
        if not root.exists():
            return pl.DataFrame(schema=_polars_schema(dataset_info)).lazy()
        lf = pl.scan_parquet((root / "**/*.parquet").as_posix())
        if filters:
            for key, value in filters.items():
                lf = lf.filter(pl.col(key) == value)
        if columns is not None:
            lf = lf.select(columns)
        return lf

    def _read_duckdb(
        self,
        dataset: str,
        dataset_info: Dataset,
        filters: Mapping[str, StoreFilterValue] | None,
        columns: list[str] | None,
    ) -> pl.LazyFrame:
        root = self._dataset_root(dataset)
        if not root.exists():
            return pl.DataFrame(schema=_polars_schema(dataset_info)).lazy()
        selected_columns = columns or [field.name for field in dataset_info.schema]
        projected = ", ".join(_quote_identifier(column) for column in selected_columns)
        params: list[object] = []
        predicates: list[str] = []
        if filters:
            for key, value in filters.items():
                predicates.append(f"{_quote_identifier(key)} = ?")
                params.append(value)
        where_clause = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        query = f"SELECT {projected} FROM {_quote_identifier(dataset)}{where_clause}"
        table = self._conn.execute(query, params).to_arrow_table()
        return pl.DataFrame(pl.from_arrow(table)).lazy()

    _YEAR_SOURCES = ("date", "announce_date", "report_date")

    def _add_partition_columns(
        self, dataset: Dataset, df: pl.DataFrame
    ) -> pl.DataFrame:
        exprs: list[pl.Expr] = []
        for col in dataset.partition_cols:
            if col in df.columns:
                continue
            if col == "year":
                exprs.append(self._year_expr(df, col))
            elif col == "date_month":
                exprs.append(pl.col("date").dt.strftime("%Y-%m").alias("date_month"))
        return df.with_columns(exprs) if exprs else df

    def _year_expr(self, df: pl.DataFrame, alias: str) -> pl.Expr:
        source_col = next((s for s in self._YEAR_SOURCES if s in df.columns), None)
        if source_col is None:
            return pl.lit(None).alias(alias)
        return pl.col(source_col).dt.year().alias(alias)

    def write(self, dataset: str, df: pl.DataFrame | pl.LazyFrame) -> None:
        dataset_info = self._dataset(dataset)
        materialized = _materialize_dataframe(df)
        output = self._deduplicate_existing(
            dataset,
            self._validate_and_prepare(dataset_info, materialized).unique(
                subset=dataset_info.dedup_keys,
                keep="last",
            ),
            dataset_info,
        )
        if output.is_empty():
            return
        root = self._partition_root(dataset, output, dataset_info.partition_cols)
        root.mkdir(parents=True, exist_ok=True)
        file_path = (
            root / f"part-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}.parquet"
        )
        pq.write_table(output.to_arrow(), file_path)
        self.register_all_views()

    def sql(self, query: str) -> pl.LazyFrame:
        relation = self._conn.sql(query)
        reader = relation.arrow()
        schema = reader.schema
        batches = list(reader)
        if not batches:
            return _from_arrow_table(pa.Table.from_batches([], schema=schema)).lazy()
        return _from_arrow_table(pa.Table.from_batches(batches, schema=schema)).lazy()

    def close(self) -> None:
        self._conn.close()

    def _register_empty_view(self, dataset_name: str) -> None:
        dataset = self._dataset(dataset_name)
        select_expr = ", ".join(
            f"CAST(NULL AS {_duckdb_type(field.type)}) AS {field.name}"
            for field in dataset.schema
        )
        self._conn.execute(
            f"CREATE OR REPLACE VIEW {dataset_name} AS SELECT {select_expr} WHERE 1 = 0"
        )

    def _validate_and_prepare(self, dataset: Dataset, df: pl.DataFrame) -> pl.DataFrame:
        expected_columns = [field.name for field in dataset.schema]
        missing = [name for name in expected_columns if name not in df.columns]
        if missing:
            msg = f"Missing columns for {dataset.name}: {missing}"
            raise ValueError(msg)
        output = self._add_partition_columns(dataset, df)
        cast_exprs = [
            self._cast_expr(field.name, field.type) for field in dataset.schema
        ]
        passthrough = [
            column for column in output.columns if column not in expected_columns
        ]
        return output.with_columns(cast_exprs).select([*expected_columns, *passthrough])

    def _partition_root(
        self,
        dataset: str,
        df: pl.DataFrame,
        partition_cols: list[str],
    ) -> Path:
        root = self._dataset_root(dataset)
        for column in partition_cols:
            if column not in df.columns or df.is_empty():
                continue
            value = df.get_column(column).to_list()[0]
            root = root / f"{column}={value}"
        return root

    def _cast_expr(self, name: str, data_type: pa.DataType) -> pl.Expr:
        return pl.col(name).cast(_polars_type(data_type), strict=False)

    def _deduplicate_existing(
        self,
        dataset: str,
        df: pl.DataFrame,
        dataset_info: Dataset,
    ) -> pl.DataFrame:
        root = self._dataset_root(dataset)
        if not root.exists() or df.is_empty():
            return df

        keys = dataset_info.dedup_keys
        existing = _materialize_dataframe(self.read(dataset, columns=keys))
        if existing.is_empty():
            return df

        existing_keys = {tuple(row) for row in existing.iter_rows()}
        mask = [tuple(row) not in existing_keys for row in df.select(keys).iter_rows()]
        if not any(mask):
            return df.head(0)
        return df.filter(pl.Series(name="_keep", values=mask))


def _materialize_dataframe(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    collected = df.collect() if isinstance(df, pl.LazyFrame) else df
    if not isinstance(collected, pl.DataFrame):
        msg = "Expected DataFrame during store materialization."
        raise TypeError(msg)
    return collected


def _from_arrow_table(table: pa.Table) -> pl.DataFrame:
    frame = pl.from_arrow(table)
    if not isinstance(frame, pl.DataFrame):
        msg = "Expected DataFrame from Arrow conversion."
        raise TypeError(msg)
    return frame


def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'
