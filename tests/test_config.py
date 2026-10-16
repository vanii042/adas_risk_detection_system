from pathlib import Path

import pytest

from driveguard.config import ConfigError, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_file_loads(path):
    cfg = load_config(path)
    assert 0 < cfg.lane.roi_top_y < cfg.lane.roi_bottom_y < 1
    assert cfg.departure.clear_offset < cfg.departure.warn_offset


def test_camera_configs_override_only_the_roi():
    default, highway = load_config(CONFIG_DIR / "default.yaml"), load_config(CONFIG_DIR / "highway.yaml")
    assert highway.lane.roi_bottom_y != default.lane.roi_bottom_y
    assert highway.detector == default.detector


def test_signs_section_has_sane_defaults():
    signs = load_config(CONFIG_DIR / "default.yaml").signs
    assert signs.enabled and 0 < signs.min_votes <= signs.vote_frames


def _write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_setting_is_rejected_with_a_helpful_message(tmp_path):
    with pytest.raises(ConfigError, match=r"unknown setting.*ttc_critcal.*Valid settings"):
        load_config(_write(tmp_path, "risk:\n  ttc_critcal: 1.0\n"))   # typo for ttc_critical


def test_unknown_section_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(_write(tmp_path, "lanes:\n  roi_top_y: 0.5\n"))


def test_section_must_be_a_mapping(tmp_path):
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(_write(tmp_path, "risk: [1, 2]\n"))


def test_empty_file_gives_defaults(tmp_path):
    assert load_config(_write(tmp_path, "")).detector.imgsz == 416
