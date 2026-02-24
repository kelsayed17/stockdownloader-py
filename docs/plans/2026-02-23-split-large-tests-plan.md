# Round 15: Split Large Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the 5 largest test files (1,376 / 1,072 / 972 / 930 / 881 lines) into focused modules, targeting all test files under 800 lines.

**Architecture:** Split each file along natural logical boundaries (parser vs client, lifecycle vs slippage, filings vs statistics). Shared helpers/fixtures are duplicated in each split file (test files prioritize independence over DRY). No conftest.py extraction — keep each test file self-contained.

**Tech Stack:** Python, pytest, no new dependencies

---

### Task 1: Split test_sec_ownership_client.py (1,376 lines)

**Files:**
- Create: `tests/data/test_sec_ownership_parsers.py`
- Modify: `tests/data/test_sec_ownership_client.py`

**Split boundary:** Parser/helper tests (pure logic, no network or filesystem) go to new file. Client/cache/network tests stay.

**Move to `test_sec_ownership_parsers.py`:**
- `TestFilingDateToQuarterEnd` (lines 165-199, 11 methods)
- `TestGetXmlText` (lines 207-232, 4 methods)
- `TestParse13fXml` (lines 240-302, 9 methods)
- `TestLegacyTextParsing` (lines 1041-1144, 11 methods)

These 4 classes test pure parsing logic with no client, no filesystem, no mocking of HTTP. They need only the XML sample constants (`_SAMPLE_13F_XML`, `_SAMPLE_13F_XML_NO_NS`, `_SAMPLE_13F_XML_INVALID`) and imports from `sec_ownership_parsers`.

**Keep in `test_sec_ownership_client.py`:**
- All network/EFTS tests (TestFetchEftsPage, TestFindInfotableUrl, TestEftsTextFallback)
- All cache tests (TestOwnershipCaching, TestPerSymbolCacheDir, TestEftsQuarterSnapshotCache)
- All integration tests (TestFetchOwnershipSnapshots, TestCusipAutoResolution, TestEftsSnapshotIntegration, TestOwnershipRateLimiting)
- Module-level helpers: `_make_client`, `_mock_response`, `_SAMPLE_EFTS_RESPONSE`, `_SAMPLE_INDEX_JSON`, `_SAMPLE_INDEX_JSON_NO_INFOTABLE`

**Expected line counts:**
- `test_sec_ownership_parsers.py`: ~350 lines (4 classes + XML constants + imports)
- `test_sec_ownership_client.py`: ~1,030 lines (10 classes + helpers)

Note: The remaining file is still over 800 but that's acceptable — the cache/network tests are logically cohesive and shouldn't be artificially split further.

**Tests:**
1. Run: `python3 -m pytest tests/data/test_sec_ownership_parsers.py tests/data/test_sec_ownership_client.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: split parser tests from test_sec_ownership_client`

### Task 2: Split test_pattern_discovery.py (1,072 lines)

**Files:**
- Create: `tests/analysis/test_pattern_discovery_filter.py`
- Modify: `tests/analysis/test_pattern_discovery.py`

**Split boundary:** Filter/statistics algorithm tests go to new file. Data model + catalog/miner tests stay.

**Move to `test_pattern_discovery_filter.py`:**
- `TestFilterPatterns` (lines 210-454, **19 methods** — largest class)
- `TestFDRCorrection` (lines 461-517, 6 methods)
- `TestDeduplication` (lines 524-559, 4 methods)
- `TestFilterPatternsTimeframe` (lines 1018-1072, 2 methods)

These 4 classes test the core filter/statistics algorithms. They need `_make_features()` and `_make_discovered()` helpers, plus imports from `pattern_discovery`.

**Keep in `test_pattern_discovery.py`:**
- Data model tests (TestPatternOutcome, TestPatternStats, TestFilterConfig)
- Catalog/miner tests (TestPatternCatalog, TestPatternMiner, TestGenerateLabel, TestConfirmationAnalysis)
- Enhancement/serialization tests (TestDiscoveredPatternEnhancement)

**Expected line counts:**
- `test_pattern_discovery_filter.py`: ~450 lines (4 classes + helpers)
- `test_pattern_discovery.py`: ~640 lines (8 classes + helpers)

**Tests:**
1. Run: `python3 -m pytest tests/analysis/test_pattern_discovery_filter.py tests/analysis/test_pattern_discovery.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: split filter/FDR tests from test_pattern_discovery`

### Task 3: Split test_intraday_backtest_engine.py (972 lines)

**Files:**
- Create: `tests/backtest/test_intraday_backtest_slippage.py`
- Modify: `tests/backtest/test_intraday_backtest_engine.py`

**Split boundary:** Slippage model tests (self-contained, already at bottom of file) go to new file. Core engine lifecycle tests stay.

**Move to `test_intraday_backtest_slippage.py`:**
- `TestSlippage` (lines 799-972, **10 methods, 173 lines**)

Also move `TestValidation` (lines 636-659, 4 methods) since it tests input validation which is a distinct concern from P/L math.

Both need the shared helpers block: `_d()`, `_make_bar()`, `_make_bars()`, `_MockStrategy`, signal factories. Copy these to the new file.

**Keep in `test_intraday_backtest_engine.py`:**
- All lifecycle tests (Long, Short, ShortLoss, ForceClose, NoDoubleEntry, EdgeCases)
- All financial mechanics tests (Commission, EquityCurveLong, EquityCurveShort, PositionSizing, ZeroNegativeRisk, InitialCapitalPreservation)

**Expected line counts:**
- `test_intraday_backtest_slippage.py`: ~320 lines (2 classes + helpers)
- `test_intraday_backtest_engine.py`: ~660 lines (12 classes + helpers)

**Tests:**
1. Run: `python3 -m pytest tests/backtest/test_intraday_backtest_slippage.py tests/backtest/test_intraday_backtest_engine.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: split slippage tests from test_intraday_backtest_engine`

### Task 4: Split test_gme_analyzer.py (930 lines)

**Files:**
- Create: `tests/analysis/test_gme_analyzer_statistics.py`
- Modify: `tests/analysis/test_gme_analyzer.py`

**Split boundary:** Price/volume statistical analysis tests go to new file. Filing/options analysis tests stay.

**Move to `test_gme_analyzer_statistics.py`:**
- `TestComputePriceStatistics` (lines 322-395, 6 methods)
- `TestDetectVolatilityRegimes` (lines 574-651, 4 methods)
- `TestAnalyzeReturnDistribution` (lines 658-751, 5 methods)
- `TestComputeVolumeProfile` (lines 758-808, 4 methods)
- `TestDetectStructuralBreaks` (lines 816-930, 5 methods)

Also move the mid-file helpers `_make_daily_series()` and `_make_benchmark_from_stock()` (lines 403-459) since they're used by the statistical tests. Copy `_bar()` and `_D` helper to the new file.

**Keep in `test_gme_analyzer.py`:**
- `TestCorrelateFilingsWithPrice` (115-186)
- `TestDetectKeyPeriods` (193-229)
- `TestAnalyzeOptionsChain` (237-315)
- `TestRunEventStudy` (467-566)
- Helpers: `_bar()`, `_filing()`, `_contract()`, `_make_chain()`, `_D`

**Expected line counts:**
- `test_gme_analyzer_statistics.py`: ~480 lines (5 classes + helpers)
- `test_gme_analyzer.py`: ~470 lines (4 classes + helpers)

**Tests:**
1. Run: `python3 -m pytest tests/analysis/test_gme_analyzer_statistics.py tests/analysis/test_gme_analyzer.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: split statistics tests from test_gme_analyzer`

### Task 5: Split test_sec_insider_client.py (881 lines)

**Files:**
- Create: `tests/data/test_sec_insider_parsers.py`
- Modify: `tests/data/test_sec_insider_client.py`

**Split boundary:** Parser-only tests (pure logic, no client/cache) go to new file. Client/cache/model tests stay.

**Move to `test_sec_insider_parsers.py`:**
- `TestPrevQuarter` (lines 88-97, 3 methods)
- `TestBulkZipParsing` (lines 104-166, 6 methods)
- `TestRealBulkZipFormat` (lines 210-254, 4 methods) — with the real-format TSV constants (lines 174-207)
- `TestNormalizeDate` (lines 262-281, 6 methods)
- `TestExtract13d13gData` (lines 306-343, 5 methods)
- `TestXmlText` (lines 566-575, 3 methods)
- `TestForm345XmlParsing` (lines 638-689, 3 methods) — with associated XML sample constants

These need the TSV sample constants and `_make_bulk_zip()` helper. No client import needed.

**Keep in `test_sec_insider_client.py`:**
- `TestInsiderCaching` (350-558, 8 methods) — the largest remaining class
- `TestCusipAutoResolution` (697-721, 2 methods)
- `TestIpoDateNarrowing` (728-750, 1 method)
- `TestInsiderTransactionModel` (758-831, 4 methods)
- `TestBeneficialOwnerModel` (834-881, 3 methods)

**Expected line counts:**
- `test_sec_insider_parsers.py`: ~450 lines (7 classes + TSV constants + helpers)
- `test_sec_insider_client.py`: ~440 lines (5 classes + helpers)

**Tests:**
1. Run: `python3 -m pytest tests/data/test_sec_insider_parsers.py tests/data/test_sec_insider_client.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: split parser tests from test_sec_insider_client`

### Task 6: Verify test counts and full regression

**Steps:**

1. Verify total test count hasn't changed:
```bash
python3 -m pytest tests/ -x -q
```
Expected: 3297 passed (same as before)

2. Verify new test files run independently:
```bash
python3 -m pytest tests/data/test_sec_ownership_parsers.py -v --tb=short | tail -3
python3 -m pytest tests/analysis/test_pattern_discovery_filter.py -v --tb=short | tail -3
python3 -m pytest tests/backtest/test_intraday_backtest_slippage.py -v --tb=short | tail -3
python3 -m pytest tests/analysis/test_gme_analyzer_statistics.py -v --tb=short | tail -3
python3 -m pytest tests/data/test_sec_insider_parsers.py -v --tb=short | tail -3
```

3. Verify no test files over 800 lines remain in the top 5:
```bash
wc -l tests/data/test_sec_ownership_client.py \
      tests/data/test_sec_ownership_parsers.py \
      tests/analysis/test_pattern_discovery.py \
      tests/analysis/test_pattern_discovery_filter.py \
      tests/backtest/test_intraday_backtest_engine.py \
      tests/backtest/test_intraday_backtest_slippage.py \
      tests/analysis/test_gme_analyzer.py \
      tests/analysis/test_gme_analyzer_statistics.py \
      tests/data/test_sec_insider_client.py \
      tests/data/test_sec_insider_parsers.py
```

---

## Verification

```bash
python3 -m pytest tests/ -x -q
find tests/ -name "*.py" -not -name "__init__.py" -exec wc -l {} + | sort -rn | head -15
```
