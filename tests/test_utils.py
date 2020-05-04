from chesty import utils


def test_resolve_symbols():
    conf = {
        'dir': '%p/../test',
        'data': 'test/%l/test',
        'l': ['%l', '/test/%p'],
        'nested': {
            'yo': '%p/test',
            'l2': ['test/%l', '%p']
        },
        'm': '%p/%l/test'
    }
    expected = {
        'dir': 'asd/../test',
        'data': 'test/qwe/test',
        'l': ['qwe', '/test/asd'],
        'nested': {
            'yo': 'asd/test',
            'l2': ['test/qwe', 'asd']
        },
        'm': 'asd/qwe/test'
    }

    symbols = {'p': 'asd', 'l': 'qwe'}
    resolved = utils.resolve_symbols(conf, symbols)

    assert resolved == expected
