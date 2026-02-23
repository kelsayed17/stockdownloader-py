# Massive Overhaul Phase 3: Deduplication & Monolith Splitting

## Context

After 9 rounds of spring cleaning (Rounds 1-9) covering all 8 packages and
cross-cutting fixes, plus the Insider Ownership feature implementation, the
codebase still has:

- **19 files over 600 lines** — monoliths needing splitting
- **SEC client pattern duplication** — 3 clients + 2 parsers sharing rate
  limiting, ZIP caching, split adjustments, quarter iteration
- **FINRA client pattern duplication** — 3 clients sharing OAuth2 auth,
  API patterns, JSON parsing
- **10 test files over 800 lines** — could be split for clarity

## Rounds

### Round 10: SEC Client Deduplication

**Problem**: `sec_insider_client.py` (836 lines), `sec_ownership_client.py`
(644 lines), and `sec_ftd_client.py` (612 lines) independently implement:
rate limiting, ZIP download + caching, split adjustments, quarter iteration
with IPO-aware floor, and retry logic.  The two parser files
(`sec_insider_parsers.py` 767 lines, `sec_ownership_parsers.py` 713 lines)
share XML parsing helpers and date utilities.

**Plan**:
1. Extract `data/sec_base_client.py` — common SEC client base with
   `download_and_cache_zip()`, `iterate_quarters()`,
   `apply_split_adjustments()`
2. Extract `data/sec_parser_utils.py` — shared XML parsing helpers,
   date formatting, CUSIP/CIK lookups
3. Refactor all 3 SEC clients to inherit from the new base
4. Refactor both parser files to use shared utilities

**Expected**: ~300-500 lines shared code extracted, each client slimmed
~100-200 lines.

### Round 11: FINRA Client Deduplication

**Problem**: `finra_dark_pool_client.py` (460 lines),
`finra_short_interest_client.py` (400 lines), and
`finra_short_volume_client.py` (574 lines) independently implement OAuth2
authentication, API query patterns, JSON response parsing, error handling.

**Plan**:
1. Extract `data/finra_base_client.py` — common OAuth2 auth, token
   management, API query wrapper, error handling
2. Refactor all 3 FINRA clients to inherit from the new base

**Expected**: ~150-250 lines shared code extracted, each client slimmed
~80-150 lines.

### Round 12: Split Monoliths — util/

**Problem**: `pinescript/generator.py` (1,211 lines) and
`indicators/hub.py` (893 lines) are the two largest source files.

**Plan**:
1. Split `generator.py` into: core generator, strategy renderer,
   composite renderer, output formatter (~300 lines each)
2. Split `hub.py` into: core IndicatorHub, cache management, streaming
   accumulator base

### Round 13: Split Monoliths — data/

**Problem**: `regsho_sources.py` (684 lines) has 4 exchange query functions
crammed together.  Post-Round 10, SEC parsers may still be large.

**Plan**:
1. Split `regsho_sources.py` by exchange: NYSE, Nasdaq, CBOE, OCC sources
2. Split large parsers if still >500 lines post-Round 10

### Round 14: Split Monoliths — ml/ & analysis/

**Problem**: `trainer.py` (786), `stage_backtest.py` (658),
`feature_extractor.py` (602), `options_gamma_analyzer.py` (748),
`value_screener.py` (662).

**Plan**:
1. Split `trainer.py` — model training vs cross-validation/hyperparameter
2. Split `options_gamma_analyzer.py` — gamma exposure, max pain, options flow
3. Split `value_screener.py` — screening criteria vs scoring logic
4. Split remaining if warranted after inspection

### Round 15: Split Large Tests

**Problem**: 10 test files over 800 lines (biggest: 1,376 lines).

**Plan**:
1. Split top 4-5 largest test files along logical boundaries

## Sequencing

| Round | Focus | Dependencies |
|-------|-------|--------------|
| 10 | SEC client dedup | None |
| 11 | FINRA client dedup | None |
| 12 | Split util/ monoliths | None |
| 13 | Split data/ monoliths | After Round 10 |
| 14 | Split ml/ & analysis/ | None |
| 15 | Split large tests | After Rounds 10-14 |

## Verification

Each round: `python3 -m pytest tests/ -x -q` must pass with 3226+ tests,
0 failures.
