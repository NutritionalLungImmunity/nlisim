from tempfile import TemporaryFile


def test_save_state(state):
    with TemporaryFile('wb') as f:
        state.save(f)
        assert f.tell() > 0


def test_serialize_state(state):
    serialized = state.serialize()
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


def test_load_state(state):
    state.concentration[:] = 1
    new_state = state.load(state.serialize())
    assert new_state is not state
    assert (new_state.concentration == state.concentration).all()
