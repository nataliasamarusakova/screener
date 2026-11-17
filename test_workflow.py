from pathlib import Path


def test_workflow_persists_jsonl_on_current_branch_without_state_branch():
    workflow = (Path(__file__).resolve().parent / ".github/workflows/screener.yml").read_text()
    assert "Persist scanner state on the current branch" in workflow
    assert 'git push origin "HEAD:${GITHUB_REF}"' in workflow
    assert "refs/heads/screener-state" not in workflow
    assert "Restore persistent analysis history" not in workflow
    assert "DATA_DIR: data" in workflow
    assert "actions/checkout@v4" in workflow
    assert "fetch-depth: 1" in workflow
    assert "fetch-tags: false" in workflow
    assert "sqlite" not in workflow.lower()


def test_data_jsonl_is_not_ignored():
    gitignore = (Path(__file__).resolve().parent / ".gitignore").read_text()
    assert "data/*.jsonl" not in gitignore


def test_workflow_caches_playwright_browser_and_pinned_wireproxy():
    workflow = (Path(__file__).resolve().parent / ".github/workflows/screener.yml").read_text()
    assert "uses: actions/cache@v4" in workflow
    assert "~/.cache/ms-playwright" in workflow
    assert "~/.cache/wireproxy" in workflow
    assert "hashFiles('requirements.txt')" in workflow
    assert "if: steps.cache-runtime.outputs.cache-hit != 'true'" in workflow
    assert "if: steps.cache-runtime.outputs.cache-hit == 'true'" in workflow
    assert workflow.count("python -m playwright install chromium") == 1
    assert "https://fapi.binance.com/fapi/v1/ping" in workflow
    assert "https://ipinfo.io/json" not in workflow
    assert "OUTPUT_DIR" not in workflow
    assert "analysis.json" not in workflow



def test_workflow_queues_pending_runs_and_does_not_commit_per_run_journal():
    root = Path(__file__).resolve().parent
    workflow = (root / ".github/workflows/screener.yml").read_text()
    gitignore = (root / ".gitignore").read_text()
    assert "queue: single" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "git add data/snapshots.jsonl data/market_rows.jsonl data/notifications.jsonl" in workflow
    assert 'if [[ -f data/poll_state.json ]]; then' in workflow
    assert 'git add data/poll_state.json' in workflow
    assert "git add data/snapshots.jsonl data/market_rows.jsonl data/notifications.jsonl data/runs.jsonl" not in workflow
    assert "touch data/snapshots.jsonl data/market_rows.jsonl data/notifications.jsonl" in workflow
    assert "git rm --cached -f data/runs.jsonl" in workflow
    assert "data/runs.jsonl" in gitignore
    assert "python -m screener.merge_state --source" in workflow
    assert "for attempt in 1 2 3 4 5" in workflow
    assert 'git fetch --no-tags origin "$GITHUB_REF"' in workflow
    assert 'git reset --hard FETCH_HEAD' in workflow
    assert 'if: always()' in workflow
    assert 'if: always() && !cancelled()' not in workflow
    assert 'timeout-minutes: 15' in workflow
    assert 'timeout-minutes: 25' in workflow
    assert 'COINALYZE_TIMEOUT_MS: "45000"' in workflow
    assert 'COINALYZE_PAGE_ATTEMPTS: "3"' in workflow
    assert 'COINALYZE_RETRY_BACKOFF_SEC: "3"' in workflow



def test_runtime_requirements_are_exactly_pinned_in_one_file():
    root = Path(__file__).resolve().parent
    requirements = (root / "requirements.txt").read_text().splitlines()
    assert requirements == [
        "beautifulsoup4==4.15.0",
        "lxml==6.1.3",
        "playwright==1.62.0",
        "playwright-stealth==2.0.3",
        "requests==2.34.2",
        "pytest==8.4.2",
    ]
    assert not (root / "requirements-dev.txt").exists()
    assert not (root / "constraints.txt").exists()



def test_workflow_prunes_after_remote_and_runner_state_are_merged():
    workflow = (Path(__file__).resolve().parent / ".github/workflows/screener.yml").read_text()
    merge_at = workflow.index("python -m screener.merge_state --source")
    prune_at = workflow.index('Store(Path("data"), create_files=False).prune', merge_at)
    stage_at = workflow.index("git add data/snapshots.jsonl", merge_at)
    assert merge_at < prune_at < stage_at



def test_workflow_runs_regressions_before_scanning_and_skips_persistence_if_tests_fail():
    workflow = (Path(__file__).resolve().parent / ".github/workflows/screener.yml").read_text()
    tests_at = workflow.index("- name: Run full regression suite")
    scan_at = workflow.index("python -m screener\n", tests_at)
    persist_at = workflow.index("- name: Persist scanner state on the current branch")
    assert tests_at < scan_at < persist_at
    assert "if: always() && steps.tests.outcome == 'success'" in workflow[persist_at:]

def test_persist_step_python_heredoc_terminator_is_at_shell_column_zero():
    workflow = (Path(__file__).resolve().parent / ".github/workflows/screener.yml").read_text()
    persist = workflow.split("- name: Persist scanner state on the current branch", 1)[1]
    lines = persist.splitlines()
    command_index = next(i for i, line in enumerate(lines) if line.strip() == "python - <<'PY'")
    command_indent = len(lines[command_index]) - len(lines[command_index].lstrip(" "))
    terminator_index = next(i for i in range(command_index + 1, len(lines)) if lines[i].strip() == "PY")
    terminator_indent = len(lines[terminator_index]) - len(lines[terminator_index].lstrip(" "))
    assert terminator_indent == command_indent
