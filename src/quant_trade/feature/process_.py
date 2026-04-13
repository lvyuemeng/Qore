from typing import Literal, Protocol

import polars as pl


@staticmethod
def _safe_div(
	num: str | pl.Expr,
	den: str | pl.Expr,
	alias: str,
	*,
	min_den: float = 1e-6,
	handle_neg_den: Literal["null", "abs", "keep"] = "null",
	fill_null: float | None = None,
) -> pl.Expr:
	n = pl.col(num) if isinstance(num, str) else num
	d = pl.col(den) if isinstance(den, str) else den

	neg_mask = d <= 0

	d_adj = d.abs() if handle_neg_den == "abs" else d
	d_safe = d_adj.clip(lower_bound=min_den)

	ratio = n / d_safe

	if handle_neg_den == "null":
		ratio = pl.when(neg_mask).then(None).otherwise(ratio)

	if fill_null is not None:
		ratio = ratio.fill_null(fill_null)

	return ratio.alias(alias)


class MetricProvider(Protocol):
    @property
    def inputs(self) -> list[str]: ...
    @property
    def outputs(self) -> list[str]: ...

    def stages(self) -> list[list[pl.Expr]]:
        """
        Returns a list of stages. 
        Stage 1 can use columns produced by Stage 0.
        """
        ...


class MetricEngine:
    def __init__(
        self,
        providers: list[MetricProvider],
        ident_cols: list[str],
        mode: Literal["append", "select"] = "append"
    ):
        self.providers = providers
        self.ident_cols = ident_cols
        self.mode = mode

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        max_stages = max(len(p.stages()) for p in self.providers)
        result = df.lazy() # Use LazyFrame for better plan optimization

        for stage_idx in range(max_stages):
            current_stage_exprs = []
            for p in self.providers:
                p_stages = p.stages()
                if stage_idx < len(p_stages):
                    current_stage_exprs.extend(p_stages[stage_idx])

            if current_stage_exprs:
                result = result.with_columns(current_stage_exprs)

        if self.mode == "select":
            all_outputs = [out for p in self.providers for out in p.outputs]
            result = result.select([*self.ident_cols, *all_outputs])

        return result.collect()
