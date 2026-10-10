"""Unit tests for FamaFrenchLoader."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from quantcore.data.sources.fama_french import FamaFrenchLoader


def _make_zip_bytes(csv_content: str) -> bytes:
    """Create a synthetic zip file containing a single CSV."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.csv", csv_content)
    return buf.getvalue()


def _sample_csv() -> str:
    """Return a minimal Ken French-like 5-factor CSV."""
    lines = [
        "Dummy header line 1",
        "Dummy header line 2",
        ",Mkt-RF,SMB,HML,RMW,CMA,RF",
        "20240102,0.05,-0.02,0.01,0.00,0.03,0.02",
        "20240103,-0.01,0.03,0.00,0.01,-0.02,0.02",
        "20240104,0.02,0.01,-0.01,0.00,0.01,0.02",
    ]
    return "\n".join(lines)


class TestFamaFrenchLoader:
    """Tests for FamaFrenchLoader."""

    def test_init_creates_cache_dir(self, tmp_path: Path) -> None:
        loader = FamaFrenchLoader(cache_dir=str(tmp_path / "ff_test"))
        assert loader._cache_dir.exists()
        assert loader._is_temp is False

    def test_init_temp_dir(self) -> None:
        loader = FamaFrenchLoader(cache_dir=None)
        assert loader._cache_dir.exists()
        assert loader._is_temp is True

    def test_parse_csv_returns_expected_columns(self) -> None:
        loader = FamaFrenchLoader(cache_dir=".")
        csv_text = _sample_csv()
        buf = io.BytesIO(csv_text.encode("utf-8"))
        df = loader._parse_csv(buf, "5_factors_2x3", "daily")
        assert list(df.columns) == [
            "Mkt-RF",
            "SMB",
            "HML",
            "RMW",
            "CMA",
            "RF",
        ]
        assert len(df) == 3
        np.testing.assert_allclose(df["Mkt-RF"].values, np.asarray([0.0005, -0.0001, 0.0002]), atol=1e-6)

    def test_parse_zip(self) -> None:
        loader = FamaFrenchLoader(cache_dir=".")
        csv_text = _sample_csv()
        zip_bytes = _make_zip_bytes(csv_text)
        df = loader._parse_zip(zip_bytes, "5_factors_2x3", "daily")
        assert not df.empty
        assert "Mkt-RF" in df.columns

    @patch("requests.get")
    def test_download_and_cache(self, mock_get: MagicMock, tmp_path: Path) -> None:
        csv_text = _sample_csv()
        zip_bytes = _make_zip_bytes(csv_text)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_bytes
        mock_get.return_value = mock_response

        cache_dir = tmp_path / "ff_cache"
        loader = FamaFrenchLoader(cache_dir=str(cache_dir))
        df = loader._download("5_factors_2x3", "daily")
        assert not df.empty
        mock_get.assert_called_once()

    def test_get_factor_returns(self, tmp_path: Path) -> None:
        loader = FamaFrenchLoader(cache_dir=str(tmp_path / "ff_cache"))
        csv_text = _sample_csv()
        # Prime cache by writing to the expected cache file
        cache_file = loader._cache_dir / "5_factors_2x3_daily.csv"
        lines = [
            "Date,Mkt-RF,SMB,HML,RMW,CMA,RF",
            "2024-01-02,0.0005,-0.0002,0.0001,0.0,0.0003,0.0002",
            "2024-01-03,-0.0001,0.0003,0.0,0.0001,-0.0002,0.0002",
            "2024-01-04,0.0002,0.0001,-0.0001,0.0,0.0001,0.0002",
        ]
        cache_file.write_text("\n".join(lines))

        arr = loader.get_factor_returns(freq="daily")
        assert arr.shape == (3, 5)
        assert arr.dtype == np.float64
