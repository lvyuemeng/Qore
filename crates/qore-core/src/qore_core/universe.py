from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterator, Sequence, ValuesView
from datetime import date
from typing import TypeVar

from qore_core.instrument import SessionInstrument, TradingSession

TInstrument = TypeVar("TInstrument", bound=SessionInstrument)


class Universe[TInstrument: SessionInstrument]:
    def __init__(self, instruments: Sequence[TInstrument]) -> None:
        if not instruments:
            self._instrument_type: type[object] | None = None
        else:
            types = {type(inst) for inst in instruments}
            if len(types) > 1:
                msg = f"Universe must be homogeneous; got {types}"
                raise TypeError(msg)
            self._instrument_type = type(instruments[0])
        self._map = {inst.symbol: inst for inst in instruments}
        self._suspended: dict[tuple[str, date], bool] = {}

    def symbols(self) -> list[str]:
        return list(self._map)

    def values(self) -> ValuesView[TInstrument]:
        return self._map.values()

    def items(self) -> ItemsView[str, TInstrument]:
        return self._map.items()

    def get(self, symbol: str) -> TInstrument:
        return self._map[symbol]

    def __iter__(self) -> Iterator[TInstrument]:
        return iter(self._map.values())

    def __len__(self) -> int:
        return len(self._map)

    @property
    def instrument_type(self) -> type[object] | None:
        return self._instrument_type

    @property
    def session(self) -> TradingSession | None:
        if not self._map:
            return None
        return next(iter(self._map.values())).session

    def is_suspended(self, symbol: str, d: date) -> bool:
        return self._suspended.get((symbol, d), False)

    def set_suspended(self, symbol: str, d: date, suspended: bool = True) -> None:
        if symbol not in self._map:
            msg = f"Unknown symbol: {symbol}"
            raise KeyError(msg)
        self._suspended[(symbol, d)] = suspended

    def copy(self) -> Universe[TInstrument]:
        copied = Universe(list(self._map.values()))
        copied._suspended = self._suspended.copy()
        return copied

    def pipe(
        self, fn: Callable[[Universe[TInstrument]], Universe[TInstrument]]
    ) -> Universe[TInstrument]:
        return fn(self)

    def filtered(
        self,
        predicate: Callable[[TInstrument], bool],
    ) -> Universe[TInstrument]:
        filtered = Universe([inst for inst in self if predicate(inst)])
        filtered._suspended = self._suspended.copy()
        return filtered

    def with_suspensions(
        self,
        suspension_state: dict[tuple[str, date], bool],
    ) -> Universe[TInstrument]:
        updated = self.copy()
        for key, value in suspension_state.items():
            symbol, suspension_date = key
            if symbol in updated._map:
                updated._suspended[(symbol, suspension_date)] = value
        return updated

    def tradeable_on(self, d: date) -> Universe[TInstrument]:
        return self.filtered(lambda inst: not self.is_suspended(inst.symbol, d))
