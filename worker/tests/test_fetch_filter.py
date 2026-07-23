from datetime import datetime
from types import SimpleNamespace

from worker.pipeline.fetch import _newest_year_items


def item(year: int, ident: str):
    return SimpleNamespace(id=ident, datetime=datetime(year, 6, 15))


def test_multiple_years_keeps_newest_only():
    items = [item(2022, "sw22"), item(2020, "sw20"), item(2018, "sw18")]
    assert [i.id for i in _newest_year_items(items)] == ["sw22"]


def test_same_year_adjacent_quads_all_kept():
    items = [item(2022, "sw22"), item(2022, "se22"), item(2020, "sw20")]
    assert [i.id for i in _newest_year_items(items)] == ["sw22", "se22"]
