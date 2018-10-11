from .. import utils


def test_func_source_data():
    filename, line, source = utils.func_source_data(test_func_source_data)

    assert filename.endswith('test_utils.py')
    assert isinstance(line, int)
    assert 'assert isinstance(line, int)' in source
