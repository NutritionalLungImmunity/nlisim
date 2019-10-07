from pathlib import Path

from simulation.state import State

_last_save = 0.0


def save_state(state: State):
    global _last_save

    save_interval = state.config.getfloat('simulation.save', 'save_interval', fallback=0.0)
    save_file_name = state.config.get(
        'simulation.save', 'save_file_name', fallback='output/simulation-%%010.3f.pkl')
    now = state.time

    if now - _last_save + 1e-8 > save_interval:
        path = Path(save_file_name % now)
        path.parent.mkdir(parents=True, exist_ok=True)
        state.save(path)
        _last_save = now

    return state
