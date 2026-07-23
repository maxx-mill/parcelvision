from worker.pipeline.fetch import _newest_year_only


def test_multiple_years_keeps_newest_only():
    tiles = [
        "/t/m_3809022_sw_15_060_20220618.tif",
        "/t/m_3809022_sw_15_060_20200705.tif",
        "/t/m_3809022_sw_15_1_20180820.tif",
        "/t/m_3809022_sw_15_060_20160710.tif",
    ]
    assert _newest_year_only(tiles) == ["/t/m_3809022_sw_15_060_20220618.tif"]


def test_same_year_adjacent_quads_all_kept():
    tiles = [
        "/t/m_3809022_sw_15_060_20220618.tif",
        "/t/m_3809022_se_15_060_20220618.tif",
    ]
    assert _newest_year_only(tiles) == tiles


def test_unparseable_names_left_alone():
    tiles = ["/t/custom_mosaic.tif"]
    assert _newest_year_only(tiles) == tiles
