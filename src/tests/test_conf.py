import src.utils.conf as conf


def test_configuration_json_consistency():
    config = conf.get_conf()
    json_1 = config.to_json()
    json_2 = conf.Configuration.from_json(json_1).to_json()
    assert json_1 == json_2
