# qore-intelligence

Model pipeline, signal generation, and registry for Qore.

## Key exports

| Name | Kind | Description |
|---|---|---|
| `ModelPipeline` | class | Training/inference orchestration |
| `ModelRegistry` | class | Model artifact store and retrieval |
| `MultiHorizonRanker` | class | LGBM ranking model |
| `build_ranking_strategy` | function | Adapts ranker to runner strategy |
| `RobustScaler` | class | Robust scaling normalization |
| `CrossSectionalZScore` | class | Cross-sectional z-score normalization |
| `RankScaler` | class | Rank-based normalization |
| `NewsPipeline` | class | News ingestion and scoring pipeline |
| `FinBERT` | class | Financial sentiment scoring |
| `LLMExtractor` | class | LLM-based event extraction |
| `Triage` | class | Signal triage/routing |
| `SignalCombiner` | class | Combine multiple signal sources |
| `IntelligenceSettings` | `@dataclass` | Model store paths |

## Pattern

```python
from qore_intelligence import IntelligenceSettings
from qore_intelligence.model.lgbm_rank import MultiHorizonRanker
from qore_intelligence.strategy import build_ranking_strategy

settings = IntelligenceSettings(model_store_root="models")
ranker = MultiHorizonRanker()
strategy = build_ranking_strategy(ranker)
```
