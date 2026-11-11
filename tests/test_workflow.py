from pathlib import Path


def test_workflow_persists_jsonl_on_current_branch_without_state_branch():
    workflow = (Path(__file__).parents[1] / ".github/workflows/screener.yml").read_text()
    assert "Persist JSONL history on the current branch" in workflow
    assert 'git push origin "HEAD:${GITHUB_REF}"' in workflow
    assert "refs/heads/screener-state" not in workflow
    assert "Restore persistent analysis history" not in workflow
    assert "DATA_DIR: data" in workflow
    assert "actions/checkout@v4" in workflow
    assert "fetch-depth: 1" in workflow
    assert "fetch-tags: false" in workflow
    assert "sqlite" not in workflow.lower()


def test_data_jsonl_is_not_ignored():
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text()
    assert "data/*.jsonl" not in gitignore


def test_workflow_caches_playwright_browser_and_pinned_wireproxy():
    workflow = (Path(__file__).parents[1] / ".github/workflows/screener.yml").read_text()
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
    root = Path(__file__).parents[1]
    workflow = (root / ".github/workflows/screener.yml").read_text()
    gitignore = (root / ".gitignore").read_text()
    assert "queue: single" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "git add data/snapshots.jsonl data/market_rows.jsonl data/notifications.jsonl" in workflow
    assert "git add data/snapshots.jsonl data/market_rows.jsonl data/notifications.jsonl data/runs.jsonl" not in workflow
    assert "git rm --cached -f data/runs.jsonl" in workflow
    assert "data/runs.jsonl" in gitignore


def test_dev_dependencies_keep_pytest_available_without_runtime_ci_workflow():
    root = Path(__file__).parents[1]
    runtime_requirements = (root / "requirements.txt").read_text()
    dev_requirements = (root / "requirements-dev.txt").read_text()
    workflow = (root / ".github/workflows/screener.yml").read_text()

    assert "pytest" not in runtime_requirements.lower()
    assert "-r requirements.txt" in dev_requirements
    assert "pytest" in dev_requirements.lower()
    assert "python -m pytest" not in workflow
    assert "python -m pip install -r requirements.txt" in workflow


def test_repository_keeps_only_runtime_workflow_and_excludes_reference_docs():
    root = Path(__file__).parents[1]
    workflows = sorted((root / ".github/workflows").glob("*.yml"))
    assert [path.name for path in workflows] == ["screener.yml"]
    assert not (root / "P1_REFERENCE.txt").exists()
    assert not (root / "OPERATIONS.md").exists()

