from pathlib import Path

import attr

from simulation.module import Module, ModuleState
from simulation.state import State


class FileOutput(Module):
    name = 'file_output'
    defaults = {
        'save_interval': '1',
        'save_file_name': 'output/simulation-<time>.pkl'
    }

    @attr.s(kw_only=True)
    class StateClass(ModuleState):
        last_save: float = attr.ib(default=0)

    def advance(self, state: State, previous_time: float) -> State:
        save_interval = self.config.getfloat('save_interval')
        save_file_name = self.config.get('save_file_name')
        now = state.time

        if now - state.file_output.last_save > save_interval - 1e-8:
            path = Path(save_file_name.replace('<time>', ('%010.3f' % now).strip()))
            path.parent.mkdir(parents=True, exist_ok=True)
            state.save(path)
            state.file_output.last_save = now

        return state
