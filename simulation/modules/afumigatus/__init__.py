"""
Afumigatus Module
"""
import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.modules.afumigatus.afumigatus_cell import AfumigatusCell
from simulation.state import State


@attr.s(kw_only=True, repr=False)
class AfumigatusState(ModuleState):
    lastId: int = attr.ib(default=-1)
    min_iter_to_status_change: int = attr.ib(default=0)
    pr_status_change: float = attr.ib(default=0.5)

    # def __repr__(self):
    #     return f'AfumigatusState(concentration, wind_x, wind_y, source)'
    pass


class Afumigatus(Module):
    name = 'afumigatus'
    defaults = {
        'trees': 'np.array([])',
        'min_iter_to_status_change': '5',
        'pr_status_change': '0.05',
        'num_spore': '0',
        'lastId': '-1'
    }

    StateClass = AfumigatusState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus

        c = self.config

        afumigatus.trees = np.array([])
        afumigatus.min_iter_to_status_change = c.getint('min_iter_to_status_change')
        afumigatus.pr_status_change = c.getfloat('pr_status_change')
        afumigatus.branch_probability = c.getfloat('branch_probability')
        afumigatus.init_spores = c.getfloat('init_spores')
        afumigatus.num_spore = 0
        afumigatus.last_id = -1

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        afumigatus: AfumigatusState = state.afumigatus

        # load afumigatus data from state
        num_spore = afumigatus.num_spore
        init_spores = afumigatus.init_spores
        trees = afumigatus.trees

        grid: State.RectangularGrid = state.grid
        xbin = grid.shape[2]
        ybin = grid.shape[1]
        zbin = grid.shape[0]

        # #iterate over roots and then sub hypheal trees
        for root in trees:
            nodes_to_process = np.array([])
            nodes_to_process = np.append(nodes_to_process, root)

            while(len(nodes_to_process) > 0):
                curr_af = nodes_to_process[0]
                nodes_to_process = np.delete(nodes_to_process, 0, axis=0)
                elongate_children = curr_af.elongate_children
                branch_children = curr_af.branch_children

                if(len(elongate_children) > 0):
                    for c in elongate_children:
                        nodes_to_process = np.append(nodes_to_process, c)

                if(len(branch_children) > 0):
                    for c in branch_children:
                        nodes_to_process = np.append(nodes_to_process, c)

                curr_af.update_status(afumigatus.pr_status_change,
                                      afumigatus.min_iter_to_status_change)

                # grow, branch
                # TODO incorporate geometry growth restriction
                new_af = curr_af.elongate(xbin, ybin, zbin)
                if(new_af):
                    new_af.id = afumigatus.last_id + 1
                    afumigatus.last_id += 1
                    afumigatus.num_spore += 1
                # TODO incorporate geometry growth restriction
                new_af2 = curr_af.branch(afumigatus.branch_probability, xbin, ybin, zbin)
                if(new_af2):
                    new_af2.id = afumigatus.last_id + 1
                    afumigatus.last_id += 1
                    afumigatus.num_spore += 1
                if(curr_af.previous_septa is None and not(curr_af.switched) and
                   len(curr_af.elongate_children) > 0):
                    # we are at root, so can grow opposite direction
                    # TODO add growth delay based on random() < prob_dual elongate
                    curr_af.dx = curr_af.dx * - 1
                    curr_af.dy = curr_af.dy * - 1
                    curr_af.dz = curr_af.dz * - 1
                    curr_af.growable = True
                    curr_af.switched = True

        # #add new spores if haven't inhaled the limit
        if(num_spore < init_spores):
            # TODO add limitation to just lodge layer eg:
            # rz, ry, rx = np.where(geometry==SPORE_LAYER) # macro read from config
            while(afumigatus.num_spore < init_spores):
                # index = randint(0, len(rz))
                afumigatus.last_id += 1
                last_id = afumigatus.last_id
                af = AfumigatusCell(x=10, y=10, z=10, ironPool=0,
                                    status=AfumigatusCell.Status.RESTING_CONIDIA,
                                    state=AfumigatusCell.State.FREE, isRoot=True, id_in=last_id)
                afumigatus.trees = np.append(afumigatus.trees, af)
                afumigatus.num_spore += 1

        # TODO diffuseIron()

        return state
