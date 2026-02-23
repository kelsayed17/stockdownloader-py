# Round 11: FINRA Client Deduplication — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract duplicated patterns (OAuth2 auth, init, rate limiting, date normalization) from 3 FINRA clients into a shared `FinraBaseClient` base class.

**Architecture:** Create `finra_base_client.py` with OAuth2 + common init. Move `_normalize_date()` to existing `sec_parser_utils.py` (it's a general date utility). Remove `_rate_limit()` overrides (base class provides it). Refactor all 3 clients to inherit from `FinraBaseClient`.

**Tech Stack:** Python, requests, base64, BaseDataClient

---

### Task 1: Create `FinraBaseClient` and refactor all 3 FINRA clients

**Files:**
- Create: `src/stockdownloader/data/finra_base_client.py`
- Modify: `src/stockdownloader/data/finra_dark_pool_client.py`
- Modify: `src/stockdownloader/data/finra_short_interest_client.py`
- Modify: `src/stockdownloader/data/finra_short_volume_client.py`
- Create: `tests/data/test_finra_base_client.py`

**Steps:**

1. Create `finra_base_client.py` (~80 lines) containing:
   - `_FINRA_TOKEN_URL` constant (currently duplicated in all 3 files)
   - `_MAX_RETRIES = 3` and `_RATE_LIMIT_DELAY = 0.5` (currently duplicated)
   - `FinraBaseClient(BaseDataClient)` class with:
     - `__init__(client_id, client_secret, data_dir)` — calls super() with
       shared headers, stores credentials from args or env vars
     - `_authenticate() -> bool` — the OAuth2 token acquisition logic
       (identical across all 3 clients)
   - Do NOT override `_rate_limit()` — let BaseDataClient handle it
   - Module-level `_normalize_finra_date(raw: str) -> str` — the shared
     date normalization (or reuse `sec_parser_utils.normalize_date` if it
     handles the same formats)

2. Refactor all 3 FINRA clients:
   - Change inheritance from `BaseDataClient` to `FinraBaseClient`
   - Remove duplicated `__init__`, `_authenticate()`, `_rate_limit()`,
     `_normalize_date()` from each
   - Replace with simple `super().__init__(client_id, client_secret, data_dir)`
   - Remove duplicated constants (`_FINRA_TOKEN_URL`, `_MAX_RETRIES`,
     `_RATE_LIMIT_DELAY`)
   - Keep client-specific methods: `_query_api()`, `_raw_to_records()`,
     `_load_cache()`, `_save_cache()`, and public fetch methods

3. Create `tests/data/test_finra_base_client.py` with tests for:
   - `test_authenticate_success` — mock token endpoint, verify token set
   - `test_authenticate_no_credentials` — returns False
   - `test_authenticate_failure` — non-200 response returns False
   - `test_normalize_date` variants (if using local function)

4. Run: `python3 -m pytest tests/data/test_finra_dark_pool_client.py tests/data/test_finra_short_interest_client.py tests/data/test_finra_short_volume_client.py tests/data/test_finra_base_client.py -x -q`

5. Run full regression: `python3 -m pytest tests/ -x -q`

6. Commit:
   ```bash
   git add src/stockdownloader/data/finra_base_client.py \
           src/stockdownloader/data/finra_dark_pool_client.py \
           src/stockdownloader/data/finra_short_interest_client.py \
           src/stockdownloader/data/finra_short_volume_client.py \
           tests/data/test_finra_base_client.py
   git commit -m "refactor: extract FinraBaseClient with shared OAuth2 + init"
   ```

### Task 2: Update exports and verify

**Files:**
- Check: `src/stockdownloader/data/__init__.py`
- Run: full test suite

**Steps:**

1. Check if `data/__init__.py` needs to export `FinraBaseClient`
2. Run: `python3 -m pytest tests/ -x -q`
3. Verify line count reductions
4. Commit if needed

---

## Verification

```bash
python3 -m pytest tests/ -x -q
wc -l src/stockdownloader/data/finra_*.py
```
