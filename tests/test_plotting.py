from dendritic_dr.plotting import config_label, display_name


def test_display_names():
    assert display_name("vanilla") == "Dense ANN"
    assert display_name("unknown-model") == "unknown-model"


def test_config_labels_from_csv_strings():
    assert config_label({"model": "dann-r", "somas": "[128]", "hidden": ""}) == "dANN (random), 128 somas"
    assert config_label({"model": "vanilla", "somas": float("nan"), "hidden": "[64]"}) == "Dense ANN, 64 hidden"
    assert config_label({"model": "dann-lrf", "somas": "[256, 64]"}) == "dANN (local RF), 256-64 somas"


def test_human_count():
    from dendritic_dr.plotting import _human_count

    assert _human_count(20000) == "20k"
    assert _human_count(200000) == "200k"
    assert _human_count(1500000) == "1.5M"
    assert _human_count(500) == "500"


def test_config_label_shows_nondefault_synapses():
    assert config_label({"model": "dann-r", "somas": "[128]", "synapses": 128}) == "dANN (random), 128 somas, 128 synapses"
    assert config_label({"model": "dann-r", "somas": "[128]", "synapses": 16.0}) == "dANN (random), 128 somas"
