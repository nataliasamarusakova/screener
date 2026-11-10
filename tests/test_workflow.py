from pathlib import Path


def test_workflow_persists_jsonl_on_current_branch_without_state_branch():
    workflow = (Path(__file__).parents[1] / ".github/workflows/screener.yml").read_text()
    assert "Persist JSONL history on the current branch" in workflow
    assert 'git push origin "HEAD:${GITHUB_REF}"' in workflow
    assert "refs/heads/screener-state" not in workflow
    assert "Restore persistent analysis history" not in workflow
    assert "DATA_DIR: data" in workflow
    assert "actions/checkout@v4" in workflow
    assert "fetch-depth: 0" in workflow
    assert "sqlite" not in workflow.lower()


def test_data_jsonl_is_not_ignored():
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text()
    assert "data/*.jsonl" not in gitignore


def test_workflow_validates_wireguard_keys_without_logging_key_material():
    workflow = (Path(__file__).parents[1] / ".github/workflows/screener.yml").read_text()
    assert "base64.b64decode(value, validate=True)" in workflow
    assert "must decode to 32 bytes" in workflow
    assert "Raw parser output is suppressed to protect keys." in workflow
    assert "--configtest -c" in workflow


def test_wireproxy_health_check_targets_binance_only_and_avoids_step_local_env_bug():
    workflow = (Path(__file__).parents[1] / ".github/workflows/screener.yml").read_text()
    assert "https://fapi.binance.com/fapi/v1/ping" in workflow
    assert "https://ipinfo.io/json" not in workflow
    assert "VPN_EXPECTED_COUNTRY" not in workflow
    assert 'os.environ["VPN_DIR"]' not in workflow
    assert 'echo "VPN_DIR=$vpn_dir" >> "$GITHUB_ENV"' not in workflow
    assert '--proxy "$BINANCE_HTTP_PROXY"' in workflow
    assert "Binance API connectivity through WireProxy: OK" in workflow
