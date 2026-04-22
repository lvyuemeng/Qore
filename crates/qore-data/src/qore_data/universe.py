from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import polars as pl
from qore_core.instrument import StockInstrument
from qore_core.universe import Universe

from qore_data.source import StockSource
from qore_data.store.duckdb import QoreStore

CandidateOperator = Literal["gt", "ge", "lt", "le", "eq", "ne", "between", "in"]
StockSnapshotKind = Literal["profile", "forecast", "announcement", "audit_opinion"]
SelectionStage = Literal[
    "profiles",
    "statuses",
    "fundamentals",
    "forecasts",
    "daily_market",
    "announcements",
]
type BetweenValue = tuple[object, object] | list[object]


@dataclass(frozen=True, slots=True)
class CandidateFilter:
    field: str
    operator: CandidateOperator
    value: object | BetweenValue
    fill_null: float | bool | str | None = None

    def expr(self) -> pl.Expr:
        column = pl.col(self.field)
        if self.fill_null is not None:
            column = column.fill_null(self.fill_null)
        value_expr = pl.lit(self.value)
        match self.operator:
            case "gt":
                return column.gt(value_expr)
            case "ge":
                return column.ge(value_expr)
            case "lt":
                return column.lt(value_expr)
            case "le":
                return column.le(value_expr)
            case "eq":
                return column.eq(value_expr)
            case "ne":
                return column.ne(value_expr)
            case "between":
                if not isinstance(self.value, tuple | list) or len(self.value) != 2:
                    msg = (
                        "'between' candidate filter requires a two-item tuple or list."
                    )
                    raise TypeError(msg)
                lower, upper = self.value[0], self.value[1]
                return column.gt(pl.lit(lower)) & column.le(pl.lit(upper))
            case "in":
                values = self.value
                if not isinstance(values, Sequence) or isinstance(values, str):
                    msg = "'in' candidate filter requires a non-string sequence value."
                    raise TypeError(msg)
                return column.is_in(list(values))


@dataclass(frozen=True, slots=True)
class CandidateSort:
    field: str
    descending: bool = False


@dataclass(frozen=True, slots=True)
class StockCandidateSpec:
    filters: tuple[CandidateFilter, ...] = field(default_factory=tuple)
    sort_by: tuple[CandidateSort, ...] = field(default_factory=tuple)
    top_n: int | None = None
    exclude_st: bool = True
    exclude_suspended: bool = True
    exclude_limit_up: bool = False
    exclude_limit_down: bool = False
    min_listing_days: int | None = None

    def required_stages(self) -> tuple[SelectionStage, ...]:
        required = {"statuses"}
        for column in self._fields_in_use():
            required.add(_selection_stage_for_field(column))
        return tuple(stage for stage in _SELECTION_STAGE_ORDER if stage in required)

    def apply_lazy(self, selection_frame: pl.LazyFrame) -> pl.LazyFrame:
        lf = selection_frame.filter(pl.col("is_tradeable").fill_null(False))
        if self.exclude_st:
            lf = lf.filter(~pl.col("is_st").fill_null(False))
        if self.exclude_suspended:
            lf = lf.filter(~pl.col("is_suspended").fill_null(False))
        if self.exclude_limit_up:
            lf = lf.filter(~pl.col("limit_up").fill_null(False))
        if self.exclude_limit_down:
            lf = lf.filter(~pl.col("limit_down").fill_null(False))
        if self.min_listing_days is not None:
            lf = lf.filter(pl.col("listing_days").fill_null(0) >= self.min_listing_days)
        for candidate_filter in self.filters:
            lf = lf.filter(candidate_filter.expr())
        if self.sort_by:
            lf = lf.sort(
                [candidate_sort.field for candidate_sort in self.sort_by],
                descending=[
                    candidate_sort.descending for candidate_sort in self.sort_by
                ],
                nulls_last=True,
            )
        if self.top_n is not None:
            lf = lf.head(self.top_n)
        return lf

    def apply(self, selection_frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        lf = (
            selection_frame.lazy()
            if isinstance(selection_frame, pl.DataFrame)
            else selection_frame
        )
        return pl.DataFrame(self.apply_lazy(lf).collect())

    def to_universe(
        self,
        selection_frame: pl.DataFrame,
        *,
        as_of: date,
        keep_suspended: bool = False,
    ) -> Universe[StockInstrument]:
        selected = self.apply(selection_frame)
        universe = Universe(
            [
                StockInstrument.from_mapping(row)
                for row in selected.iter_rows(named=True)
            ]
        )
        if keep_suspended and "is_suspended" in selected.columns:
            for symbol in (
                selected.filter(pl.col("is_suspended").fill_null(False))
                .get_column("symbol")
                .to_list()
            ):
                universe.set_suspended(str(symbol), as_of)
        return universe

    def _fields_in_use(self) -> set[str]:
        fields = {candidate_filter.field for candidate_filter in self.filters}
        fields.update(candidate_sort.field for candidate_sort in self.sort_by)
        if self.min_listing_days is not None:
            fields.add("listing_days")
        return fields


@dataclass(frozen=True, slots=True)
class StockSelectionScope:
    index_symbol: str
    as_of: date
    announcement_start: date | None = None
    announcement_end: date | None = None


@dataclass(frozen=True, slots=True)
class StockSelectionPipeline:
    store: QoreStore
    scope: StockSelectionScope
    frame: pl.LazyFrame
    stages: frozenset[SelectionStage] = field(default_factory=frozenset)

    @classmethod
    def from_index(
        cls,
        store: QoreStore,
        *,
        index_symbol: str,
        as_of: date,
        announcement_start: date | None = None,
        announcement_end: date | None = None,
    ) -> StockSelectionPipeline:
        scope = StockSelectionScope(
            index_symbol=index_symbol,
            as_of=as_of,
            announcement_start=announcement_start,
            announcement_end=announcement_end,
        )
        frame = store.read_duckdb(
            "index_constituents",
            filters={"index_symbol": index_symbol, "as_of": as_of},
        ).with_columns(pl.lit(as_of).alias("selection_date"))
        return cls(store=store, scope=scope, frame=frame)

    def with_stage(self, stage: SelectionStage) -> StockSelectionPipeline:
        if stage == "profiles":
            return self.with_profiles()
        if stage == "statuses":
            return self.with_statuses()
        if stage == "fundamentals":
            return self.with_fundamentals()
        if stage == "forecasts":
            return self.with_forecasts()
        if stage == "daily_market":
            return self.with_daily_market()
        return self.with_announcement_counts()

    def with_stages(self, *stages: SelectionStage) -> StockSelectionPipeline:
        pipeline = self
        for stage in stages:
            pipeline = pipeline.with_stage(stage)
        return pipeline

    def with_default_selection_inputs(self) -> StockSelectionPipeline:
        return self.with_stages(*_SELECTION_STAGE_ORDER)

    def with_candidate_inputs(
        self,
        candidate_spec: StockCandidateSpec,
    ) -> StockSelectionPipeline:
        return self.with_stages(*candidate_spec.required_stages())

    def with_category_inputs(self) -> StockSelectionPipeline:
        return self.with_stages("profiles", "forecasts", "announcements")

    def candidates(self, candidate_spec: StockCandidateSpec) -> pl.DataFrame:
        return candidate_spec.apply(
            self.with_candidate_inputs(candidate_spec).frame,
        )

    def to_universe(
        self,
        candidate_spec: StockCandidateSpec,
        *,
        keep_suspended: bool = False,
    ) -> Universe[StockInstrument]:
        return candidate_spec.to_universe(
            self.with_candidate_inputs(candidate_spec).collect(),
            as_of=self.scope.as_of,
            keep_suspended=keep_suspended,
        )

    def category_report(self) -> pl.DataFrame:
        return pl.DataFrame(
            self.with_category_inputs()
            .frame.group_by("industry", "board")
            .agg(
                pl.len().alias("symbol_count"),
                pl.col("total_market_cap").mean().alias("avg_total_market_cap"),
                pl.col("report_count").mean().alias("avg_report_count"),
                pl.col("announcement_count").sum().alias("announcement_count"),
            )
            .sort("industry")
            .collect()
        )

    def collect(self) -> pl.DataFrame:
        return pl.DataFrame(self.frame.collect())

    def selection_frame(self) -> pl.DataFrame:
        frame = self.with_default_selection_inputs().collect()
        if frame.is_empty():
            return _empty_selection_frame()
        return frame.sort("total_market_cap", nulls_last=True)

    def with_profiles(self) -> StockSelectionPipeline:
        if "profiles" in self.stages:
            return self
        profiles = self.store.read_duckdb(
            "stock_profiles",
            filters={"as_of": self.scope.as_of},
        )
        frame = self.frame.join(
            profiles, on=["symbol", "as_of"], how="left"
        ).with_columns(
            pl.coalesce("exchange_right", "exchange").alias("exchange"),
            pl.coalesce("industry_right", "industry").alias("industry"),
            pl.when(pl.col("listing_date").is_not_null())
            .then((pl.lit(self.scope.as_of) - pl.col("listing_date")).dt.total_days())
            .otherwise(None)
            .alias("listing_days"),
        )
        return self._replace(
            frame=frame.drop("exchange_right", "industry_right"),
            stage="profiles",
        )

    def with_statuses(self) -> StockSelectionPipeline:
        pipeline = self.with_profiles()
        if "statuses" in pipeline.stages:
            return pipeline
        daily = pipeline.store.read_duckdb(
            "stock_ohlcv",
            filters={"date": pipeline.scope.as_of},
            columns=["symbol", "is_suspended"],
        ).rename({"is_suspended": "daily_is_suspended"})
        frame = pipeline.frame.join(daily, on="symbol", how="left").with_columns(
            pl.col("daily_is_suspended").fill_null(False).alias("is_suspended"),
            pl.when(pl.col("is_st").fill_null(False))
            .then(0.05)
            .otherwise(0.10)
            .alias("price_limit_pct"),
        )
        frame = frame.with_columns(
            (
                ~pl.col("is_st").fill_null(False)
                & ~pl.col("is_suspended").fill_null(False)
            ).alias("is_tradeable")
        )
        return pipeline._replace(
            frame=frame.drop("daily_is_suspended"), stage="statuses"
        )

    def with_fundamentals(self) -> StockSelectionPipeline:
        if "fundamentals" in self.stages:
            return self
        return self._replace(
            frame=self.frame.join(
                _latest_fundamentals_lazy(self.store, as_of=self.scope.as_of),
                on="symbol",
                how="left",
            ),
            stage="fundamentals",
        )

    def with_forecasts(self) -> StockSelectionPipeline:
        if "forecasts" in self.stages:
            return self
        return self._replace(
            frame=self.frame.join(
                self.store.read_duckdb(
                    "analyst_forecasts",
                    filters={"as_of": self.scope.as_of},
                ),
                on=["symbol", "as_of"],
                how="left",
            ),
            stage="forecasts",
        )

    def with_daily_market(self) -> StockSelectionPipeline:
        if "daily_market" in self.stages:
            return self
        daily = self.store.read_duckdb(
            "stock_ohlcv",
            filters={"date": self.scope.as_of},
            columns=["symbol", "amount", "limit_up", "limit_down"],
        )
        frame = self.frame.join(daily, on="symbol", how="left").with_columns(
            pl.col("amount").fill_null(0.0),
            pl.col("limit_up").fill_null(False),
            pl.col("limit_down").fill_null(False),
        )
        return self._replace(frame=frame, stage="daily_market")

    def with_announcement_counts(self) -> StockSelectionPipeline:
        if "announcements" in self.stages:
            return self
        frame = self.frame.join(
            _announcement_counts_lazy(
                self.store,
                start=self.scope.announcement_start,
                end=self.scope.announcement_end,
            ),
            on="symbol",
            how="left",
        ).with_columns(pl.col("announcement_count").fill_null(0))
        return self._replace(frame=frame, stage="announcements")

    def with_audit_opinion_state(
        self,
        *,
        adverse_codes: tuple[str, ...] = ("disclaimer", "adverse"),
        max_age_days: int | None = None,
    ) -> StockSelectionPipeline:
        state = _latest_audit_opinion_state_lazy(
            self.store,
            as_of=self.scope.as_of,
            adverse_codes=adverse_codes,
            max_age_days=max_age_days,
        )
        frame = self.frame.join(state, on="symbol", how="left").with_columns(
            pl.col("has_adverse_audit_opinion").fill_null(False),
            pl.col("active_audit_exclusion").fill_null(False),
        )
        return StockSelectionPipeline(
            store=self.store,
            scope=self.scope,
            frame=frame,
            stages=self.stages,
        )

    def _replace(
        self,
        *,
        frame: pl.LazyFrame,
        stage: SelectionStage,
    ) -> StockSelectionPipeline:
        return StockSelectionPipeline(
            store=self.store,
            scope=self.scope,
            frame=frame,
            stages=frozenset((*self.stages, stage)),
        )


@dataclass(frozen=True, slots=True)
class StockSnapshotSpec:
    dataset: Literal[
        "stock_profiles",
        "analyst_forecasts",
        "announcements",
        "stock_audit_opinions",
    ]
    kind: StockSnapshotKind
    schema: Mapping[str, pl.DataType]

    async def fetch_frames(
        self,
        source: StockSource,
        instruments: Sequence[StockInstrument],
        *,
        as_of: date | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> list[pl.DataFrame]:
        if self.kind == "profile":
            if as_of is None:
                msg = "Profile snapshot requires as_of."
                raise ValueError(msg)
            return [await source.stock_profile(inst, as_of) for inst in instruments]
        if self.kind == "forecast":
            if as_of is None:
                msg = "Forecast snapshot requires as_of."
                raise ValueError(msg)
            return [await source.analyst_forecast(inst, as_of) for inst in instruments]
        if self.kind == "audit_opinion":
            if start is None or end is None:
                msg = "Audit opinion snapshot requires start and end."
                raise ValueError(msg)
            return [
                await source.audit_opinions(inst, start, end) for inst in instruments
            ]
        if start is None or end is None:
            msg = "Announcement snapshot requires start and end."
            raise ValueError(msg)
        return [await source.announcements(inst, start, end) for inst in instruments]

    async def snapshot(
        self,
        source: StockSource,
        store: QoreStore,
        *,
        instruments: Sequence[StockInstrument],
        as_of: date | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        frames = await self.fetch_frames(
            source,
            instruments,
            as_of=as_of,
            start=start,
            end=end,
        )
        non_empty = [frame for frame in frames if not frame.is_empty()]
        if not non_empty:
            return pl.DataFrame(schema=self.schema)
        combined = pl.concat(non_empty, how="vertical")
        store.write(self.dataset, combined)
        return combined


_STOCK_PROFILES_SCHEMA = {
    "as_of": pl.Date(),
    "symbol": pl.String(),
    "short_name": pl.String(),
    "exchange": pl.String(),
    "industry": pl.String(),
    "board": pl.String(),
    "listing_date": pl.Date(),
    "total_market_cap": pl.Float64(),
    "float_market_cap": pl.Float64(),
    "total_shares": pl.Float64(),
    "float_shares": pl.Float64(),
    "is_st": pl.Boolean(),
}

_ANALYST_FORECASTS_SCHEMA = {
    "as_of": pl.Date(),
    "symbol": pl.String(),
    "report_count": pl.Int64(),
    "buy": pl.Int64(),
    "overweight": pl.Int64(),
    "neutral": pl.Int64(),
    "underweight": pl.Int64(),
    "sell": pl.Int64(),
    "eps_year1": pl.Float64(),
    "eps_year2": pl.Float64(),
    "eps_year3": pl.Float64(),
    "eps_year4": pl.Float64(),
}

_ANNOUNCEMENTS_SCHEMA = {
    "symbol": pl.String(),
    "short_name": pl.String(),
    "title": pl.String(),
    "notice_type": pl.String(),
    "notice_date": pl.Date(),
    "art_code": pl.String(),
    "url": pl.String(),
}

_AUDIT_OPINIONS_SCHEMA = {
    "symbol": pl.String(),
    "report_date": pl.Date(),
    "announce_date": pl.Date(),
    "opinion": pl.String(),
    "opinion_code": pl.String(),
    "source_notice_type": pl.String(),
    "title": pl.String(),
    "art_code": pl.String(),
    "url": pl.String(),
}

_AUDIT_OPINION_STATE_SCHEMA = {
    "symbol": pl.String(),
    "latest_audit_report_date": pl.Date(),
    "latest_audit_announce_date": pl.Date(),
    "latest_audit_opinion": pl.String(),
    "latest_audit_opinion_code": pl.String(),
    "has_adverse_audit_opinion": pl.Boolean(),
    "adverse_audit_opinion_age_days": pl.Int64(),
    "active_audit_exclusion": pl.Boolean(),
}

_FUNDAMENTAL_SELECTION_SCHEMA = {
    "symbol": pl.String(),
    "report_date": pl.Date(),
    "announce_date": pl.Date(),
    "pe_ttm": pl.Float64(),
    "pb": pl.Float64(),
    "ps_ttm": pl.Float64(),
    "ev_ebitda": pl.Float64(),
    "roe": pl.Float64(),
    "roa": pl.Float64(),
    "gross_margin": pl.Float64(),
    "revenue": pl.Float64(),
    "net_income": pl.Float64(),
    "total_liabilities": pl.Float64(),
    "total_assets": pl.Float64(),
    "operating_cashflow": pl.Float64(),
}

_EMPTY_SELECTION_FRAME_SCHEMA = {
    "selection_date": pl.Date,
    "index_symbol": pl.String,
    "symbol": pl.String,
    "exchange": pl.String,
    "industry": pl.String,
    "short_name": pl.String,
    "board": pl.String,
    "listing_date": pl.Date,
    "listing_days": pl.Int64,
    "is_st": pl.Boolean,
    "is_suspended": pl.Boolean,
    "is_tradeable": pl.Boolean,
    "price_limit_pct": pl.Float64,
    "limit_up": pl.Boolean,
    "limit_down": pl.Boolean,
    "amount": pl.Float64,
    "total_market_cap": pl.Float64,
    "float_market_cap": pl.Float64,
    "total_shares": pl.Float64,
    "float_shares": pl.Float64,
    "report_date": pl.Date,
    "announce_date": pl.Date,
    "pe_ttm": pl.Float64,
    "pb": pl.Float64,
    "ps_ttm": pl.Float64,
    "ev_ebitda": pl.Float64,
    "roe": pl.Float64,
    "roa": pl.Float64,
    "gross_margin": pl.Float64,
    "revenue": pl.Float64,
    "net_income": pl.Float64,
    "total_liabilities": pl.Float64,
    "total_assets": pl.Float64,
    "operating_cashflow": pl.Float64,
    "report_count": pl.Int64,
    "buy": pl.Int64,
    "overweight": pl.Int64,
    "neutral": pl.Int64,
    "underweight": pl.Int64,
    "sell": pl.Int64,
    "eps_year1": pl.Float64,
    "eps_year2": pl.Float64,
    "eps_year3": pl.Float64,
    "eps_year4": pl.Float64,
    "announcement_count": pl.UInt32,
}

_SELECTION_STAGE_ORDER: tuple[SelectionStage, ...] = (
    "profiles",
    "statuses",
    "fundamentals",
    "forecasts",
    "daily_market",
    "announcements",
)

_SELECTION_STAGE_BY_FIELD: dict[str, SelectionStage] = {
    "short_name": "profiles",
    "board": "profiles",
    "listing_date": "profiles",
    "listing_days": "profiles",
    "total_market_cap": "profiles",
    "float_market_cap": "profiles",
    "total_shares": "profiles",
    "float_shares": "profiles",
    "is_st": "statuses",
    "is_suspended": "statuses",
    "is_tradeable": "statuses",
    "price_limit_pct": "statuses",
    "report_date": "fundamentals",
    "announce_date": "fundamentals",
    "pe_ttm": "fundamentals",
    "pb": "fundamentals",
    "ps_ttm": "fundamentals",
    "ev_ebitda": "fundamentals",
    "roe": "fundamentals",
    "roa": "fundamentals",
    "gross_margin": "fundamentals",
    "revenue": "fundamentals",
    "net_income": "fundamentals",
    "total_liabilities": "fundamentals",
    "total_assets": "fundamentals",
    "operating_cashflow": "fundamentals",
    "report_count": "forecasts",
    "buy": "forecasts",
    "overweight": "forecasts",
    "neutral": "forecasts",
    "underweight": "forecasts",
    "sell": "forecasts",
    "eps_year1": "forecasts",
    "eps_year2": "forecasts",
    "eps_year3": "forecasts",
    "eps_year4": "forecasts",
    "amount": "daily_market",
    "limit_up": "daily_market",
    "limit_down": "daily_market",
    "announcement_count": "announcements",
}


def _selection_stage_for_field(field: str) -> SelectionStage:
    if field in {
        "selection_date",
        "index_symbol",
        "symbol",
        "exchange",
        "industry",
    }:
        return "profiles"
    if field not in _SELECTION_STAGE_BY_FIELD:
        msg = f"Unsupported stock selection field: {field}"
        raise KeyError(msg)
    return _SELECTION_STAGE_BY_FIELD[field]


async def snapshot_index_constituents(
    source: StockSource,
    store: QoreStore,
    *,
    index_symbol: str,
    as_of: date,
) -> pl.DataFrame:
    instruments = await source.index_constituents(index_symbol, as_of)
    frame = _index_constituents_frame(
        instruments, index_symbol=index_symbol, as_of=as_of
    )
    store.write("index_constituents", frame)
    return frame


async def snapshot_stock_profiles(
    source: StockSource,
    store: QoreStore,
    *,
    instruments: Sequence[StockInstrument],
    as_of: date,
) -> pl.DataFrame:
    return await StockSnapshotSpec(
        dataset="stock_profiles",
        kind="profile",
        schema=_STOCK_PROFILES_SCHEMA,
    ).snapshot(
        source,
        store,
        instruments=instruments,
        as_of=as_of,
    )


async def snapshot_stock_analyst_forecasts(
    source: StockSource,
    store: QoreStore,
    *,
    instruments: Sequence[StockInstrument],
    as_of: date,
) -> pl.DataFrame:
    return await StockSnapshotSpec(
        dataset="analyst_forecasts",
        kind="forecast",
        schema=_ANALYST_FORECASTS_SCHEMA,
    ).snapshot(
        source,
        store,
        instruments=instruments,
        as_of=as_of,
    )


async def snapshot_stock_announcements(
    source: StockSource,
    store: QoreStore,
    *,
    instruments: Sequence[StockInstrument],
    start: date,
    end: date,
) -> pl.DataFrame:
    return await StockSnapshotSpec(
        dataset="announcements",
        kind="announcement",
        schema=_ANNOUNCEMENTS_SCHEMA,
    ).snapshot(
        source,
        store,
        instruments=instruments,
        start=start,
        end=end,
    )


async def snapshot_stock_audit_opinions(
    source: StockSource,
    store: QoreStore,
    *,
    instruments: Sequence[StockInstrument],
    start: date,
    end: date,
) -> pl.DataFrame:
    return await StockSnapshotSpec(
        dataset="stock_audit_opinions",
        kind="audit_opinion",
        schema=_AUDIT_OPINIONS_SCHEMA,
    ).snapshot(
        source,
        store,
        instruments=instruments,
        start=start,
        end=end,
    )


async def build_stock_universe_from_index(
    source: StockSource,
    store: QoreStore,
    *,
    index_symbol: str,
    as_of: date,
) -> Universe[StockInstrument]:
    instruments = await source.index_constituents(index_symbol, as_of)
    constituents = _index_constituents_frame(
        instruments, index_symbol=index_symbol, as_of=as_of
    )
    store.write("index_constituents", constituents)
    profiles = await snapshot_stock_profiles(
        source,
        store,
        instruments=instruments,
        as_of=as_of,
    )
    frame = pl.DataFrame(
        constituents.lazy()
        .join(profiles.lazy(), on=["symbol", "as_of"], how="left")
        .with_columns(
            pl.coalesce("exchange_right", "exchange").alias("exchange"),
            pl.coalesce("industry_right", "industry").alias("industry"),
            pl.lit(0.10).alias("price_limit_pct"),
        )
        .drop("exchange_right", "industry_right")
        .collect()
    )
    return Universe(
        [StockInstrument.from_mapping(row) for row in frame.iter_rows(named=True)]
    )


def snapshot_stock_statuses(
    store: QoreStore,
    *,
    as_of: date,
) -> pl.DataFrame:
    profiles = store.read("stock_profiles", filters={"as_of": as_of})
    daily = (
        store.read("stock_ohlcv", filters={"date": as_of})
        .select("symbol", "is_suspended")
        .rename({"is_suspended": "daily_is_suspended"})
    )
    frame = pl.DataFrame(
        profiles.join(daily, on="symbol", how="left")
        .with_columns(
            pl.col("daily_is_suspended").fill_null(False).alias("is_suspended"),
            pl.when(pl.col("is_st"))
            .then(0.05)
            .otherwise(0.10)
            .alias("price_limit_pct"),
        )
        .with_columns(
            (~pl.col("is_st") & ~pl.col("is_suspended")).alias("is_tradeable")
        )
        .select(
            pl.col("as_of"),
            pl.col("symbol"),
            pl.col("board"),
            pl.col("industry"),
            pl.col("is_st"),
            pl.col("is_suspended"),
            pl.col("price_limit_pct"),
            pl.col("is_tradeable"),
        )
        .collect()
    )
    store.write("stock_statuses", frame)
    return frame


def _index_constituents_frame(
    instruments: Sequence[StockInstrument],
    *,
    index_symbol: str,
    as_of: date,
) -> pl.DataFrame:
    rows = [
        {
            "as_of": as_of,
            "index_symbol": index_symbol,
            "symbol": inst.symbol,
            "exchange": inst.exchange,
            "industry": inst.industry,
        }
        for inst in instruments
    ]
    return pl.DataFrame(rows)


def _latest_fundamentals_lazy(store: QoreStore, *, as_of: date) -> pl.LazyFrame:
    return (
        store.read_duckdb("fundamentals")
        .filter(pl.col("announce_date") <= as_of)
        .sort(
            ["symbol", "announce_date", "report_date"], descending=[False, True, True]
        )
        .group_by("symbol")
        .first()
    )


def _latest_fundamentals_frame(store: QoreStore, *, as_of: date) -> pl.DataFrame:
    frame = pl.DataFrame(_latest_fundamentals_lazy(store, as_of=as_of).collect())
    if frame.is_empty():
        return pl.DataFrame(schema=_FUNDAMENTAL_SELECTION_SCHEMA)
    return frame.select(list(_FUNDAMENTAL_SELECTION_SCHEMA))


def _announcement_counts_lazy(
    store: QoreStore,
    *,
    start: date | None,
    end: date | None,
) -> pl.LazyFrame:
    announcements = store.read_duckdb("announcements")
    if start is not None:
        announcements = announcements.filter(pl.col("notice_date") >= start)
    if end is not None:
        announcements = announcements.filter(pl.col("notice_date") <= end)
    return announcements.group_by("symbol").agg(pl.len().alias("announcement_count"))


def _announcement_counts(
    store: QoreStore,
    *,
    start: date | None,
    end: date | None,
) -> pl.DataFrame:
    frame = pl.DataFrame(
        _announcement_counts_lazy(store, start=start, end=end).collect()
    )
    if frame.is_empty():
        return pl.DataFrame(
            schema={"symbol": pl.String, "announcement_count": pl.UInt32}
        )
    return frame


def _latest_audit_opinion_state_lazy(
    store: QoreStore,
    *,
    as_of: date,
    adverse_codes: tuple[str, ...],
    max_age_days: int | None,
) -> pl.LazyFrame:
    opinions = store.read_duckdb("stock_audit_opinions").filter(
        pl.col("announce_date") <= as_of
    )
    active_expr = pl.col("has_adverse_audit_opinion")
    if max_age_days is not None:
        active_expr = active_expr & (
            pl.col("adverse_audit_opinion_age_days") <= max_age_days
        )
    latest = (
        opinions.sort(
            ["symbol", "announce_date", "report_date"],
            descending=[False, True, True],
        )
        .group_by("symbol")
        .first()
        .rename(
            {
                "report_date": "latest_audit_report_date",
                "announce_date": "latest_audit_announce_date",
                "opinion": "latest_audit_opinion",
                "opinion_code": "latest_audit_opinion_code",
            }
        )
        .with_columns(
            pl.col("latest_audit_opinion_code")
            .is_in(list(adverse_codes))
            .alias("has_adverse_audit_opinion"),
            pl.when(pl.col("latest_audit_announce_date").is_not_null())
            .then(
                (pl.lit(as_of) - pl.col("latest_audit_announce_date")).dt.total_days()
            )
            .otherwise(None)
            .alias("adverse_audit_opinion_age_days"),
        )
        .with_columns(active_expr.alias("active_audit_exclusion"))
        .select(list(_AUDIT_OPINION_STATE_SCHEMA))
    )
    return latest


def _empty_selection_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_EMPTY_SELECTION_FRAME_SCHEMA)
