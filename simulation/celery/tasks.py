from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import attr
from celery import Task
from girder_client import GirderClient

from simulation.celery.app import app
from simulation.config import SimulationConfig
from simulation.postprocess import generate_vtk
from simulation.solver import run_iterator


@attr.s(auto_attribs=True, kw_only=True)
class GirderConfig:
    """Configure where the data from a simulation run is posted."""

    #: authentication token
    token: str

    #: root folder id where the data will be placed
    folder: str

    #: base api url
    api: str = 'https://data.nutritionallungimmunity.org/api/v1'

    @property
    def client(self) -> GirderClient:
        cl = GirderClient(apiUrl=self.api)
        cl.token = self.token
        return cl

    def upload(self, name: str, directory: Path) -> str:
        """Upload files to girder and return the created folder id."""
        client = self.client
        folder = client.createFolder(self.folder, name)['_id']
        for file in directory.glob('*'):
            self.client.uploadFileToFolder(folder, str(file))
        return folder


@app.task(bind=True)
def run_simulation(
    self: Task, girder_config: GirderConfig, simulation_config: SimulationConfig, target_time: float
) -> List[str]:
    """Run a simulation and export postprocessed vtk files to girder."""

    def update_task_state(status: str):
        if not self.request.called_directly:
            meta = {
                'time_step': time_step,
                'current_time': state.time,
                'target_time': target_time,
                'folders': folders,
            }
            self.update_state(state='PENDING', meta=meta)

    folders: List[str] = []
    time_step = 0

    update_task_state('PENDING')
    for state in run_iterator(simulation_config, target_time):
        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            generate_vtk(state, temp_dir_path)

            if state.time < target_time - 1e-8:
                name = '%03i' % time_step
            else:
                name = 'final'

            folders.append(girder_config.upload(name, temp_dir_path))
            update_task_state('PENDING')
        time_step += 1

    update_task_state('SUCCESS')
    return folders
