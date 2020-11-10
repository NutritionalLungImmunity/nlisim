from pathlib import Path
import shutil

from attr import attrib, attrs

from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


@attrs(kw_only=True)
class StateOutputState(ModuleState):
    last_save: float = attrib(default=0)


class StateOutput(ModuleModel):
    """
    After time steps, serialize the simulation state to an HDF5 file.

    When registered for execution, this module should always be placed last in the ordering.
    """

    name = 'state_output'

    StateClass = StateOutputState

    @property
    def _output_dir(self) -> Path:
        return Path(self.config.get('output_dir'))

    def _write_output(self, state: State) -> None:
        now = state.time
        output_file_name = f'simulation-{now:010.3f}.hdf5'
        output_file_path = self._output_dir / output_file_name

        state.save(output_file_path)

        state.state_output.last_save = now

    @staticmethod
    def _clear_directory(directory: Path) -> None:
        """Clear the contents of a directory, without removing the directory itself."""
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)

    def initialize(self, state: State) -> State:
        output_dir = self._output_dir
        if output_dir.exists():
            # Since output_dir may be a Docker mount point, don't remove the directory itself.
            print(f'File output directory {output_dir.resolve()} exists. Clearing it.')
            self._clear_directory(output_dir)
        else:
            output_dir.mkdir(parents=True)

        # Initial time is typically 0, but it can be read from state
        self._write_output(state)
        return state

    def advance(self, state: State, previous_time: float) -> State:
        self._write_output(state)
        return state
