# Pre-2013 13F Ownership — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the SEC ownership client to fetch pre-2013 13F institutional ownership data via EFTS by adding a legacy text parser that extracts holdings from non-XML filings.

**Architecture:** The existing `_fetch_from_efts()` queries EFTS by CUSIP and downloads infotable documents. Currently it only parses XML. We add `_parse_13f_text()` as a fallback when XML parsing returns empty. We also lower the `ownership_start_year` floor from 2013 to 2003 and update `_find_infotable_url()` to match `.txt` files.

**Tech Stack:** Python 3.12, pytest, `re` (stdlib regex), no new dependencies.

---

### Task 1: Lower `ownership_start_year` floor and update tests

**Files:**
- Modify: `src/stockdownloader/model/symbol_info.py:70-77`
- Modify: `tests/model/test_symbol_info.py:102-112`

**Step 1: Update the failing tests first**

In `tests/model/test_symbol_info.py`, update the three `ownership_start_year` tests.
The floor drops from 2013 to 2003.

```python
    def test_ownership_start_year_after_2003(self) -> None:
        info = _make_info(ipo_date=date(2020, 1, 1))
        assert info.ownership_start_year == 2020

    def test_ownership_start_year_before_2003(self) -> None:
        info = _make_info(ipo_date=date(1999, 2, 13))
        assert info.ownership_start_year == 2003

    def test_ownership_start_year_exactly_2003(self) -> None:
        info = _make_info(ipo_date=date(2003, 6, 1))
        assert info.ownership_start_year == 2003
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/model/test_symbol_info.py::TestProperties::test_ownership_start_year_before_2003 -v`
Expected: FAIL — currently returns 2013, test expects 2003.

**Step 3: Update `ownership_start_year` property**

In `src/stockdownloader/model/symbol_info.py`, replace lines 70-77:

```python
    @property
    def ownership_start_year(self) -> int:
        """Earliest useful year for 13F ownership data (floored at 2003).

        SEC bulk 13F data sets begin Q2 2013, but EFTS full-text search
        can find 13F-HR filings back to ~2001.  The legacy text parser
        handles pre-2013 non-XML formats.  We use 2003 as the floor
        because earlier filings are sparse and unreliable.
        """
        return max(self.ipo_date.year, 2003)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/model/test_symbol_info.py -v`
Expected: All 26 tests PASS.

**Step 5: Commit**

```bash
git add src/stockdownloader/model/symbol_info.py tests/model/test_symbol_info.py
git commit -m "feat: lower ownership_start_year floor from 2013 to 2003 for EFTS coverage"
```

---

### Task 2: Add `_parse_13f_text()` static method with tests

**Files:**
- Modify: `src/stockdownloader/data/sec_ownership_client.py` (add method after `_parse_13f_xml`)
- Modify: `tests/data/test_sec_ownership_client.py` (add test class)

**Step 1: Write the failing tests**

Add this test class at the bottom of `tests/data/test_sec_ownership_client.py`, just before the `TestOwnershipRateLimiting` class. Add `import re` if not already imported.

```python
# ------------------------------------------------------------------
# Tests: Legacy text parsing for pre-2013 13F filings
# ------------------------------------------------------------------


class TestLegacyTextParsing:
    """Tests for _parse_13f_text() — extracts holdings from non-XML tables."""

    _CUSIP = "36467W109"

    def test_tab_separated_format(self) -> None:
        """Tab-separated infotable with standard column order."""
        content = (
            "NAME OF ISSUER\tTITLE OF CLASS\tCUSIP\tVALUE\tSHRSORPRNAMT\tSH/PRN\n"
            "GAMESTOP CORP\tCOM\t36467W109\t5000\t200000\tSH\n"
            "APPLE INC\tCOM\t037833100\t999999\t50000\tSH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 200000
        assert result[0].value_usd == 5000

    def test_fixed_width_format(self) -> None:
        """Fixed-width layout typical of early 2000s filings."""
        content = (
            "GAMESTOP CORP NEW       COM        36467W109      3500       150000   SH\n"
            "MICROSOFT CORP          COM        594918104     99000      1200000   SH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 150000
        assert result[0].value_usd == 3500

    def test_comma_separated_format(self) -> None:
        """CSV-style infotable."""
        content = (
            "GAMESTOP CORP,COM,36467W109,7200,300000,SH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 300000
        assert result[0].value_usd == 7200

    def test_cusip_not_found_returns_empty(self) -> None:
        """When the CUSIP doesn't appear, return empty list."""
        content = "APPLE INC\tCOM\t037833100\t999\t50000\tSH\n"
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_empty_content_returns_empty(self) -> None:
        result = SecOwnershipClient._parse_13f_text("", self._CUSIP)
        assert result == []

    def test_xml_content_returns_empty(self) -> None:
        """XML content should not match (handled by _parse_13f_xml)."""
        content = '<?xml version="1.0"?>\n<root><cusip>36467W109</cusip></root>'
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert result == []

    def test_multiple_holders_same_cusip(self) -> None:
        """Multiple institutions holding the same CUSIP."""
        content = (
            "FUND A\tCOM\t36467W109\t1000\t50000\tSH\n"
            "FUND B\tCOM\t36467W109\t2000\t80000\tSH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 2
        total = sum(h.shares for h in result)
        assert total == 130000

    def test_cusip_case_insensitive(self) -> None:
        """CUSIP matching should be case-insensitive."""
        content = "GAMESTOP CORP\tCOM\t36467w109\t5000\t200000\tSH\n"
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1

    def test_value_with_commas_in_number(self) -> None:
        """Some filings use comma-formatted numbers like 1,500."""
        content = (
            "GAMESTOP CORP    COM    36467W109    1,500    50,000    SH\n"
        )
        result = SecOwnershipClient._parse_13f_text(content, self._CUSIP)
        assert len(result) == 1
        assert result[0].shares == 50000
        assert result[0].value_usd == 1500
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestLegacyTextParsing -v`
Expected: FAIL — `SecOwnershipClient` has no attribute `_parse_13f_text`.

**Step 3: Implement `_parse_13f_text()`**

In `src/stockdownloader/data/sec_ownership_client.py`, add `import re` at the top (after the other stdlib imports around line 28). Then add this method after the `_parse_13f_xml` method (after line 912):

```python
    # ------------------------------------------------------------------
    # Legacy text parsing (pre-2013 non-XML fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_13f_text(
        content: str,
        cusip: str,
    ) -> list[InstitutionalHolding]:
        """Parse a legacy (pre-2013) text infotable for holdings matching *cusip*.

        Pre-2013 13F filings use heterogeneous ASCII formats: tab-separated,
        comma-separated, or fixed-width.  This parser doesn't attempt full
        table parsing — it scans for lines containing the CUSIP and extracts
        integer tokens as (value, shares).

        Parameters
        ----------
        content:
            Raw text content of the information table document.
        cusip:
            9-character CUSIP to search for (case-insensitive).

        Returns
        -------
        List of :class:`InstitutionalHolding` for rows matching *cusip*.
        Returns empty list if content looks like XML or CUSIP is absent.
        """
        # Skip XML content — that's handled by _parse_13f_xml.
        stripped = content.lstrip()
        if stripped.startswith("<?xml") or stripped.startswith("<"):
            return []

        cusip_upper = cusip.upper().replace(" ", "")
        holdings: list[InstitutionalHolding] = []

        for line in content.splitlines():
            # Case-insensitive CUSIP match
            if cusip_upper not in line.upper().replace(" ", ""):
                continue

            # Extract all integer-like tokens (strip commas from numbers)
            # e.g. "1,500" -> "1500", "200000" -> "200000"
            tokens = re.findall(r"[\d,]+", line)
            integers: list[int] = []
            for tok in tokens:
                cleaned = tok.replace(",", "")
                if cleaned.isdigit() and int(cleaned) > 0:
                    integers.append(int(cleaned))

            if len(integers) < 2:
                logger.debug(
                    "Skipping line with < 2 numeric tokens: %s",
                    line[:120],
                )
                continue

            # Convention: value is reported in $1000s (smaller number),
            # shares is the actual count (larger number).
            # Sort ascending and take the two largest — but value < shares
            # for any normal holding, so min=value, max=shares.
            integers.sort()
            value_usd = integers[-2]  # second largest = value ($1000s)
            shares = integers[-1]     # largest = shares

            # Sanity: if "value" > "shares", swap — could be reversed cols
            if value_usd > shares:
                value_usd, shares = shares, value_usd

            try:
                holdings.append(InstitutionalHolding(
                    filing_date="",
                    manager_name="",
                    manager_cik="",
                    shares=shares,
                    value_usd=value_usd,
                    share_class="SH",
                ))
            except ValueError:
                continue

        return holdings
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestLegacyTextParsing -v`
Expected: All 9 tests PASS.

**Step 5: Commit**

```bash
git add src/stockdownloader/data/sec_ownership_client.py tests/data/test_sec_ownership_client.py
git commit -m "feat: add _parse_13f_text() for pre-2013 legacy 13F filing formats"
```

---

### Task 3: Modify `_find_infotable_url()` to find `.txt` files

**Files:**
- Modify: `src/stockdownloader/data/sec_ownership_client.py:805-830`
- Modify: `tests/data/test_sec_ownership_client.py` (add test)

**Step 1: Write the failing test**

Add to `tests/data/test_sec_ownership_client.py` inside a new test class after `TestLegacyTextParsing`:

```python
# ------------------------------------------------------------------
# Tests: _find_infotable_url txt fallback
# ------------------------------------------------------------------


class TestFindInfotableUrl:
    """Verify _find_infotable_url finds .txt when no .xml exists."""

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_finds_txt_when_no_xml(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """Pre-2013 filings have infotable as .txt, not .xml."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        index_json = {
            "directory": {
                "item": [
                    {"name": "primary_doc.html", "size": "1234"},
                    {"name": "infotable.txt", "size": "5678"},
                ],
            },
        }

        with patch.object(client, "_session") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = index_json
            mock_session.get.return_value = mock_resp

            result = client._find_infotable_url(
                "https://www.sec.gov/Archives/edgar/data/12345/0001234/",
            )

        assert result is not None
        assert result.endswith("/infotable.txt")

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_prefers_xml_over_txt(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """When both .xml and .txt exist, prefer .xml."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        index_json = {
            "directory": {
                "item": [
                    {"name": "infotable.xml", "size": "1234"},
                    {"name": "infotable.txt", "size": "5678"},
                ],
            },
        }

        with patch.object(client, "_session") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = index_json
            mock_session.get.return_value = mock_resp

            result = client._find_infotable_url(
                "https://www.sec.gov/Archives/edgar/data/12345/0001234/",
            )

        assert result is not None
        assert result.endswith("/infotable.xml")
```

**Step 2: Run tests to verify the txt test fails**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestFindInfotableUrl::test_finds_txt_when_no_xml -v`
Expected: FAIL — currently returns `None` for `.txt`-only filings.

**Step 3: Update `_find_infotable_url()`**

Replace lines 805-830 of `src/stockdownloader/data/sec_ownership_client.py`:

```python
    def _find_infotable_url(self, index_url: str) -> str | None:
        """Given a filing index URL, find the infotable document.

        Prefers XML (post-2013) but falls back to TXT (pre-2013).
        """
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                json_url = index_url.rstrip("/") + "/index.json"
                resp = self._session.get(json_url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("directory", {}).get("item", [])
                    # Pass 1: prefer infotable XML
                    for item in items:
                        name = item.get("name", "").lower()
                        if "infotable" in name and name.endswith(".xml"):
                            return index_url.rstrip("/") + "/" + item["name"]
                    # Pass 2: any XML that isn't the primary doc
                    for item in items:
                        name = item.get("name", "").lower()
                        if name.endswith(".xml") and "primary" not in name:
                            return index_url.rstrip("/") + "/" + item["name"]
                    # Pass 3: infotable TXT (pre-2013 fallback)
                    for item in items:
                        name = item.get("name", "").lower()
                        if "infotable" in name and name.endswith(".txt"):
                            return index_url.rstrip("/") + "/" + item["name"]
                    break
            except (
                requests.RequestException, json.JSONDecodeError, OSError,
            ):
                pass
            if attempt < _MAX_RETRIES - 1:
                time.sleep(1.0)
        return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestFindInfotableUrl -v`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/stockdownloader/data/sec_ownership_client.py tests/data/test_sec_ownership_client.py
git commit -m "feat: _find_infotable_url falls back to .txt for pre-2013 filings"
```

---

### Task 4: Wire text fallback into `_fetch_from_efts()` and lower quarter floor

**Files:**
- Modify: `src/stockdownloader/data/sec_ownership_client.py:695-699` (EFTS fallback)
- Modify: `src/stockdownloader/data/sec_ownership_client.py:256-264` (quarter floor)
- Modify: `tests/data/test_sec_ownership_client.py` (add integration test)

**Step 1: Write the failing test**

Add to `tests/data/test_sec_ownership_client.py`:

```python
# ------------------------------------------------------------------
# Tests: EFTS text fallback integration
# ------------------------------------------------------------------


class TestEftsTextFallback:
    """Verify _fetch_from_efts falls back to text parsing when XML fails."""

    @patch("stockdownloader.data.sec_ownership_client.time")
    def test_text_fallback_when_xml_empty(
        self, mock_time: MagicMock, tmp_path: Path,
    ) -> None:
        """When XML parsing returns nothing, text parsing should kick in."""
        mock_time.monotonic.return_value = 100.0
        mock_time.sleep = MagicMock()

        client = _make_client(tmp_path)

        # Simulate EFTS returning one hit
        fake_hits = [{
            "_id": "0001234-05-000001:infotable.txt",
            "_source": {
                "adsh": "0001234-05-000001",
                "ciks": ["1234"],
                "display_names": ["Test Fund"],
                "file_date": "2005-05-15",
                "period_ending": "2005-03-31",
            },
        }]

        # The downloaded content is plain text, not XML
        text_content = (
            "GAMESTOP CORP\tCOM\t36467W109\t5000\t200000\tSH\n"
        )

        with patch.object(client, "_fetch_efts_page", return_value=fake_hits), \
             patch.object(client, "_fetch_url_text", return_value=text_content):
            result = client._fetch_from_efts(
                "GME", "36467W109", 2005, 1, "2005-03-31",
            )

        assert result is not None
        assert result.total_institutional_shares == 200000
        assert result.num_institutions == 1
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestEftsTextFallback -v`
Expected: FAIL — `_parse_13f_xml` returns `[]` for text content, and there's no fallback yet.

**Step 3: Modify `_fetch_from_efts()` — add text fallback**

In `src/stockdownloader/data/sec_ownership_client.py`, replace the parsing block around lines 695-699. Find this code:

```python
            xml_content = self._fetch_url_text(xml_url)
            if xml_content is None:
                continue

            parsed = self._parse_13f_xml(xml_content, cusip)
```

Replace with:

```python
            doc_content = self._fetch_url_text(xml_url)
            if doc_content is None:
                continue

            # Try XML first (post-2013), then text fallback (pre-2013).
            parsed = self._parse_13f_xml(doc_content, cusip)
            if not parsed:
                parsed = self._parse_13f_text(doc_content, cusip)
```

**Step 4: Update the quarter floor**

In `src/stockdownloader/data/sec_ownership_client.py`, find the `min_year, min_quarter` block around lines 256-264:

```python
        # Compute the earliest useful quarter.  SEC bulk 13F data starts
        # at Q2 2013 — that is the absolute floor.  If the symbol IPO'd
        # after Q2 2013 we can skip even more.
        min_year, min_quarter = 2013, 2
```

Replace with:

```python
        # Compute the earliest useful quarter.  EFTS full-text search can
        # find 13F-HR filings back to ~2001, and the legacy text parser
        # handles pre-2013 non-XML formats.  We use Q1 2003 as the
        # absolute floor (earlier filings are sparse and unreliable).
        min_year, min_quarter = 2003, 1
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py::TestEftsTextFallback -v`
Expected: PASS.

Then run the full test suite to check for regressions:

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3005+).

**Step 6: Commit**

```bash
git add src/stockdownloader/data/sec_ownership_client.py tests/data/test_sec_ownership_client.py
git commit -m "feat: wire text parsing fallback into EFTS and lower quarter floor to Q1 2003"
```

---

### Task 5: Full suite verification

**Step 1: Run all ownership tests**

Run: `python3 -m pytest tests/data/test_sec_ownership_client.py -v`
Expected: All tests pass including the new classes.

**Step 2: Run all symbol_info tests**

Run: `python3 -m pytest tests/model/test_symbol_info.py -v`
Expected: All tests pass with updated `ownership_start_year` assertions.

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass, no regressions.

---

## Verification Checklist

After all tasks complete, verify:

1. `SymbolInfo("GME", ...).ownership_start_year` returns `2003` (not 2013)
2. `_parse_13f_text()` extracts holdings from tab, CSV, and fixed-width formats
3. `_find_infotable_url()` returns `.txt` URLs when no `.xml` exists
4. `_fetch_from_efts()` falls back to text parsing when XML returns empty
5. `fetch_ownership_snapshots("GME", num_quarters=200)` would attempt quarters back to Q1 2003
6. Full test suite passes with zero regressions
