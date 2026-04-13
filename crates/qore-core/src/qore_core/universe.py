from __future__ import annotations

from collections.abc import Iterator
from datetime import date

from qore_core.instrument import Instrument


class Universe:
    def __init__(self, instruments: list[Instrument]) -> None:
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

    def get(self, symbol: str) -> Instrument:
        return self._map[symbol]

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self._map.values())

    def __len__(self) -> int:
        return len(self._map)

    def is_suspended(self, symbol: str, d: date) -> bool:
        return self._suspended.get((symbol, d), False)

    def set_suspended(self, symbol: str, d: date, suspended: bool = True) -> None:
        if symbol not in self._map:
            msg = f"Unknown symbol: {symbol}"
            raise KeyError(msg)
        self._suspended[(symbol, d)] = suspended

    def tradeable_on(self, d: date) -> Universe:
        instruments = [
            inst for inst in self._map.values() if not self.is_suspended(inst.symbol, d)
        ]
        tradeable = Universe(instruments)
        tradeable._suspended = self._suspended.copy()
        return tradeable
