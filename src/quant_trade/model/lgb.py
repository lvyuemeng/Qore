from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TypeAlias

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from quant_trade import feature
from quant_trade.config.logger import log, debug_null_profile
from quant_trade.model.process import (
    LabelBuilder,
    DiscreteLabelBuilder,
    IdentityLabelBuilder,
    BinaryLabelBuilder,
)
from quant_trade.model.store import ModelCard,ModelMeta, ModelStore

LGBModelCard: TypeAlias = ModelCard[lgb.Booster]
LGBModel:TypeAlias = lgb.Booster

@dataclass
class ModelResult:
    model: lgb.Booster
    feature_names: list[str]
    metric_name: str
    metric_val: float
    params: dict[str, Any]
    importance: dict[str, float]

    def pack(self,name:str,*,tags:Sequence[str] = (),version:str = "0.0.1",description:str = "") -> LGBModelCard:
        meta = ModelMeta(
            name,
            feature_names=self.feature_names,
            importance=self.importance,
            metric_name=self.metric_name,
            metric_value=self.metric_val,
            tags=list(tags),
            version=version,
            framework="lightgbm",
            description=description,
        )
        return ModelCard(self.model,meta)

    def summary(self, top_k: int = 10) -> str:
        lines = []

        lines.append("=== LightGBM Training Result ===")
        lines.append(f"Metric (val): {self.metric_val:.6f}")
        lines.append("")

        lines.append("Top features by importance:")
        imp = (
            sorted(self.importance.items(), key=lambda x: x[1], reverse=True)
        )
        for name, score in imp[:top_k]:
            lines.append(f"  {name:<25} {score}")

        lines.append("")
        lines.append("Key parameters:")
        keys = TuneConfig.param_names()
        for k in keys:
            if k in self.params:
                lines.append(f"  {k:<20} {self.params[k]}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class TuneConfig:
    """
    Common training config - objective/metric agnostic.
    
    Contains: num_boost_round, seed, early_stopping, log_period, default_params.
    """
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    log_period: int = 100
    seed: int = 42
    
    default_params: dict[str, Any] = field(
        default_factory=lambda: {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "min_child_samples": 50,
        }
    )
    
    @staticmethod
    def param_names() -> list[str]:
        return [
            "num_leaves",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "min_child_samples",]


@dataclass(frozen=True)
class Processor:
    """
    Metric-specific config with static factory methods.
    
    Usage:
        metric_config = MetricConfig.ranking(
            GaussianLabelBuilder(factor="return_1m"),
            metric="ndcg@10",
        )
        
        metric_config = MetricConfig.regression(
            IdentityLabelBuilder(factor="return_1m"),
            metric="rmse",
        )
        
        metric_config = MetricConfig.binary(
            BinaryLabelBuilder(factor="signal"),
            metric="auc",
        )
    """
    
    objective: str
    metric: str
    ndcg_eval_at: tuple[int, ...] | None
    label_builder: LabelBuilder

    def metric_key(self) -> str:
        match self.metric:
            case "ndcg" | "map":
                # For ranking, LGBM appends @K using the first value in the eval_at list
                k = self.ndcg_eval_at[0] if self.ndcg_eval_at else 1
                return f"{self.metric}@{k}"
            
            case "binary_logloss" | "auc" | "rmse" | "l2" | "l1" | "mape":
                # These usually remain unchanged
                return self.metric
                
            case _:
                # Fallback for aliases or custom metrics
                return self.metric
            
    
    @staticmethod
    def ranking(
        label_builder: DiscreteLabelBuilder,
        *,
        metric: Literal["ndcg","map" ] = "ndcg",
    ) -> "Processor":
        """Create ranking config - objective='lambdarank' with discrete labels."""
        if metric.startswith("ndcg@"):
            ndcg = (int(metric.split("@")[1]),)
        else:
            ndcg = (10,)
        
        return Processor(
            objective="lambdarank",
            metric=metric,
            ndcg_eval_at=ndcg,
            label_builder=label_builder,
        )
    
    @staticmethod
    def regression(
        label_builder: IdentityLabelBuilder,
        *,
        metric: Literal["rmse", "mae", "mape"] = "rmse",
    ) -> "Processor":
        """Create regression config - objective='regression'."""
        return Processor(
            objective="regression",
            metric=metric,
            ndcg_eval_at=None,
            label_builder=label_builder,
        )
    
    @staticmethod
    def binary(
        label_builder: BinaryLabelBuilder,
        *,
        metric: Literal["binary_logloss", "binary_error", "auc"] = "binary_logloss",
    ) -> "Processor":
        """Create binary config - objective='binary'."""
        return Processor(
            objective="binary",
            metric=metric,
            ndcg_eval_at=None,
            label_builder=label_builder,
        )
    
    def lgb_params(
        self,
        optimized_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build LGB params dict - metric-specific only."""
        params = dict(optimized_params or {})
        params.update({
            "objective": self.objective,
            "metric": self.metric,
            "verbosity": -1,
        })
        
        if self.ndcg_eval_at is not None:
            params["ndcg_eval_at"] = list(self.ndcg_eval_at)
        
        return params
    
    def optimization_direction(self) -> str:
        """Get direction for Optuna optimization."""
        if "ndcg" in self.metric or "auc" in self.metric:
            return "maximize"
        return "minimize"

    def prepare(self, df: pl.DataFrame, features:list[str], *,ref: lgb.Dataset | None = None) -> tuple[lgb.Dataset, list[str]]:
        """Used by Trainer: Drops null labels and builds grouped Dataset."""
        label_builder = self.label_builder
        label_col = label_builder.label_name
        if not label_builder.factor in df.columns:
            raise ValueError(f"None of the required label {label_builder.factor} found in DataFrame")

        df = label_builder.label(df)
        avail_feats = [f for f in features if f in df.columns]
        if not avail_feats:
            raise ValueError(f"None of the required features {features[:3]}... found in DataFrame")

        df = df.filter(pl.col(label_col).is_not_null())
        if len(df) == 0:
            raise ValueError(f"Training DF is empty after dropping null labels in {label_col}")
        
        group_col = label_builder.rank_by_name
        df = df.sort(group_col)
        X = df.select(avail_feats).to_numpy()
        y = df.select(label_col).to_series().to_numpy()
        groups = (
            df.group_by(group_col, maintain_order=True)
            .len().select("len").to_series().to_numpy()
        )
        dataset = lgb.Dataset(X, y, group=groups, feature_name=avail_feats, reference=ref, free_raw_data=False)
        return dataset,avail_feats

@dataclass
class Trainer:
    """
    Trainer instantiated by Processor.
    
    Usage:
        metric_config = MetricConfig.ranking(GaussianLabelBuilder(factor="return_1m"))
        processor = Processor(features, metric_config)
        trainer = NewTrainer(processor)
        result = trainer.train(train_df, val_df, CommonConfig())
    """
    
    processor: Processor
    features: list[str]
    optimize: bool = True
    n_trials: int = 50
    
    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        common_config: TuneConfig,
    ) -> ModelResult:
        """
        Train model - metric_config comes from processor.
        
        Only CommonConfig needed - metric_config already in processor.
        """
        common = common_config
        
        # Build datasets
        processor = self.processor
        train_ds, features = processor.prepare(train_df,self.features)
        val_ds,_ = processor.prepare(val_df,self.features)
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective."""
            opt_params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            }
            
            lgb_params = processor.lgb_params(optimized_params=opt_params)
            log.debug(f"lgb params: {lgb_params}")
            model = lgb.train(
                lgb_params,
                train_set=train_ds,
                valid_sets=[train_ds,val_ds],
                valid_names=["train","valid"],
                num_boost_round=common.num_boost_round,
                callbacks=[lgb.early_stopping(common.early_stopping_rounds)],
            )
            
            return model.best_score["valid"][processor.metric_key()]
        
        # Optimization
        if self.optimize:
            sampler = optuna.samplers.TPESampler(seed=common.seed)
            study = optuna.create_study(
                direction=processor.optimization_direction(),
                sampler=sampler,
            )
            study.optimize(objective, n_trials=self.n_trials)
            optimized_params = {**common.default_params, **study.best_params}
        else:
            optimized_params = dict(common.default_params)
        
        # Final training
        lgb_params = processor.lgb_params(optimized_params=optimized_params)
        lgb_params.update({"seed": common.seed})
        
        model = lgb.train(
            lgb_params,
            train_ds,
            valid_sets=[val_ds],
            num_boost_round=common.num_boost_round,
            callbacks=[
                lgb.early_stopping(common.early_stopping_rounds),
                lgb.log_evaluation(common.log_period),
            ],
        )
        
        metric_val = model.best_score["valid_0"][processor.metric_key()]
        importance = dict(
            sorted(
                zip(features, model.feature_importance().astype(float)), 
                key=lambda x: x[1], 
                reverse=True
            )
        )
        
        return ModelResult(
            model=model,
            metric_name=processor.metric,
            feature_names=features,
            metric_val=metric_val,
            params=lgb_params,
            importance=importance,
        )
    
    def batch_train(
        self,
        batch_iter: Generator[tuple[pl.DataFrame, pl.DataFrame]],
        common_config: TuneConfig = TuneConfig(),
    ) -> ModelResult:
        """
        Train a single model using batches for memory efficiency.
        
        Usage:
            def batch_iter():
                for train_df, val_df in cv.split(df):
                    yield train_df, val_df
            
            result = trainer.train_batchwise(batch_iter, CommonConfig())
        """
        # Accumulate all training batches
        train_batches = []
        val_batches = []

        for train_df, val_df in batch_iter:
            train_batches.append(train_df)
            val_batches.append(val_df)

        # Concatenate all batches
        full_train = pl.concat(train_batches)
        full_val = pl.concat(val_batches)

        log.info(f"Total accumulated: Train={len(full_train):,}, Val={len(full_val):,}")

        # Train single model
        return self.train(
            train_df=full_train,
            val_df=full_val,
            common_config=common_config,
        )
    

@dataclass
class Predictor:
    """
    Predictor for generating inference from trained LightGBM models.
    
    Usage:
        predictor = Predictor(processor)
        predictions = predictor.predict(model_result, test_df)
        
        # Or with stored model:
        predictor = Predictor.from_stored(processor, store, model_name)
        predictions = predictor.predict_df(test_df)
    """
    features: list[str]
    model: lgb.Booster
    
    @classmethod
    def from_store(cls,name:str,*,store:ModelStore,) -> "Predictor":
        card = store.retrieve(name=name)
        return cls(features=card.meta.feature_names,model=card.model)
    
    def _prepare(self, df:pl.DataFrame) -> np.ndarray:
        X = df.select(self.features).to_numpy() 
        return X

    def predict(self,df:pl.DataFrame,score_name:str) -> pl.DataFrame:
        dataset = self._prepare(df)
        predict = self.model.predict(dataset.data,num_iteration=self.model.best_iteration)
        res = df.with_columns(pl.Series(name=score_name,values=predict))
        return res