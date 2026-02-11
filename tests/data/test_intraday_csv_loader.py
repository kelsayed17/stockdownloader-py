"""Tests for IntradayCsvLoader."""

from decimal import Decimal
from pathlib import Path

from stockdownloader.data.intraday_csv_loader import IntradayCsvLoader
from stockdownloader.model.intraday_price_data import IntradayPriceData


# Path to real 5-minute data file
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "spy_5m_bars.csv"


class TestIntradayCsvLoader:

    def test_cannot_instantiate(self):
        import pytest
        with pytest.raises(TypeError):
            IntradayCsvLoader()

    def test_load_from_real_file(self):
        if not _DATA_FILE.exists():
            import pytest
            pytest.skip("data/spy_5m_bars.csv not found")
        data = IntradayCsvLoader.load_from_file(_DATA_FILE)
        assert len(data) > 100
        assert isinstance(data[0], IntradayPriceData)
        # Check first bar has reasonable values
        assert data[0].close > Decimal("0")
        assert data[0].volume > 0

    def test_loaded_data_has_datetime_properties(self):
        if not _DATA_FILE.exists():
            import pytest
            pytest.skip("data/spy_5m_bars.csv not found")
        data = IntradayCsvLoader.load_from_file(_DATA_FILE)
        bar = data[0]
        assert len(bar.trading_date) == 10  # YYYY-MM-DD
        assert len(bar.time_str) == 8  # HH:MM:SS

    def test_load_from_missing_file(self):
        data = IntradayCsvLoader.load_from_file("/nonexistent/file.csv")
        assert data == []

    def test_load_preserves_order(self):
        if not _DATA_FILE.exists():
            import pytest
            pytest.skip("data/spy_5m_bars.csv not found")
        data = IntradayCsvLoader.load_from_file(_DATA_FILE)
        # Bars should be in chronological order
        for i in range(1, min(10, len(data))):
            assert data[i].date >= data[i - 1].date
