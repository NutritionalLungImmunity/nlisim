from enum import IntEnum
import math
from random import random, uniform

import numpy as np

from simulation.cell_lib.cell import Cell


class AfumigatusCell(Cell):
    name = 'AfumigatusCell'
    InitAfumigatusBooleanState = [
        1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # status:
    class Status(IntEnum):
        RESTING_CONIDIA = 0
        SWELLING_CONIDIA = 1
        HYPHAE = 2
        DYING = 3
        DEAD = 4

    # state:
    class State(IntEnum):
        FREE = 0
        INTERNALIZING = 1
        RELEASING = 2

    def __init__(self, x=0, y=0, z=0, iron_pool=0, status=Status.RESTING_CONIDIA,
                 state=State.FREE, is_root=True, id_in=-1):
        self.id = id_in
        self.iron_pool = iron_pool
        self.state = state
        self.status = status
        self.is_root = is_root
        self.x = x
        self.y = y
        self.z = z
        self.dx = 0.02*(uniform(-1, 1))
        self.dy = 0.02*(uniform(-1, 1))
        self.dz = 0.02*(uniform(-1, 1))

        self.growable = True
        self.switched = False
        self.branchable = False
        self.iteration = 0
        self.boolean_network = AfumigatusCell.InitAfumigatusBooleanState.copy()

        self.branch_children = np.array([])
        self.elongate_children = np.array([])
        self.previous_septa = None
        self.Fe = False

    def __str__(self):
        s = 'AfumigatusCell_' + str(self.id) + \
            ' x=' + '%.5f' % self.x + \
            ' y=' + '%.5f' % self.y + \
            ' z=' + '%.5f' % self.z + \
            ' dx=' + '%.5f' % self.dx + \
            ' dy=' + '%.5f' % self.dy + \
            ' dz=' + '%.5f' % self.dz

        #    if(self.status == AfumigatusCell.Status.RESTING_CONIDIA):
        #        s = s + ' RESTING_CONIDIA'
        #    if(self.status == AfumigatusCell.Status.SWELLING_CONIDIA):
        #        s = s + ' SWELLING_CONIDIA'
        #    if(self.status == AfumigatusCell.Status.HYPHAE):
        #        s = s + ' HYPHAE'
        #    if(self.status == AfumigatusCell.Status.DYING):
        #        s = s + ' DYING'
        #    if(self.status == AfumigatusCell.Status.DEAD):
        #        s = s + ' DEAD'
        #    if(self.state == AfumigatusCell.State.FREE):
        #        s = s + ' FREE'
        #    if(self.state == AfumigatusCell.State.INTERNALIZING):
        #        s = s + ' INTERNALIZING'
        #    if(self.state == AfumigatusCell.State.RELEASING):
        #        s = s + ' RELEASING'

        # if(self.previous_septa):
        #     s = s + ' prev=' + str(self.previous_septa.id)
        # if(len(self.branch_children) > 0):
        #     s = s + ' b_children= ' + str(self.branch_children[0].id)
        #     for c in self.branch_children[1:]:
        #         s = s + ', ' + str(c.id)
        # if(len(self.elongate_children) > 0):
        #     s = s + ' e_children= ' + str(self.elongate_children[0].id)
        #     for c in self.elongate_children[1:]:
        #         s = s + ', ' + str(c.id)
        return s

    def set_growth_vector(self, growth_vector):
        self.dx = growth_vector[0]
        self.dy = growth_vector[1]
        self.dz = growth_vector[2]

    def elongate(self, xbin, ybin, zbin):
        # TODO incorporate geometry growth restriction
        if self.growable and self.status == AfumigatusCell.Status.HYPHAE:
            # TODO add iron dependence and self.boolean_network[AfumigatusCell.LIP] == 1:
            if(self.x + self.dx < 0 or self.y + self.dy < 0 or self.z + self.dz < 0
               or self.x + self.dx > xbin or self.y + self.dy > ybin or self.z + self.dz > zbin):
                return None
            else:
                self.growable = False
                self.branchable = True
                self.iron_pool = self.iron_pool / 2.0
                child = AfumigatusCell(x=self.x + self.dx, y=self.y + self.dy, z=self.z + self.dz,
                                       iron_pool=0, status=AfumigatusCell.Status.HYPHAE,
                                       state=self.state, is_root=False)
                child.previous_septa = self
                child.iron_pool = self.iron_pool
                child.is_root = False
                self.elongate_children = np.append(self.elongate_children, child)
                return child
        return None

    def branch(self, branch_probability, xbin, ybin, zbin):
        # TODO incorporate geometry growth restriction
        if self.branchable and self.status == AfumigatusCell.Status.HYPHAE:
            # TODO add iron dependence  and self.boolean_network[AfumigatusCell.LIP] == 1:
            if random() < branch_probability:
                self.iron_pool = self.iron_pool / 2.0
                growth_vector = [self.dx, self.dy, self.dz]
                b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], growth_vector])
                b_inv = np.linalg.inv(b)
                r = AfumigatusCell.rotatation_matrix(2*random()*math.pi)
                r = np.dot(b, np.dot(r, b_inv))
                growth_vector = np.dot(r, growth_vector)

                next_x = growth_vector[0] + self.x
                next_y = growth_vector[1] + self.y
                next_z = growth_vector[2] + self.z

                if(next_x < 0 or next_y < 0 or next_z < 0
                   or next_x > xbin or next_y > ybin or next_z > zbin):
                    return None
                else:
                    child = AfumigatusCell(x=next_x, y=next_y, z=next_z,
                                           iron_pool=0, status=AfumigatusCell.Status.HYPHAE,
                                           state=self.state, is_root=False)
                    child.set_growth_vector(growth_vector)
                    child.iron_pool = self.iron_pool
                    child.previous_septa = self
                    child.is_root = False
                    self.branch_children = np.append(self.branch_children, child)

                    # self.branchable = False
                    # # set neighbors to be unbranchable
                    # self.next_branch.branchable = False
                    # if(self.previous_septa):
                    #    self.previous_septa.branchable = False

                    return child
                # self.branchable = False
        return None

    def update_status(self, probability_status_change, min_iter_to_status_change):
        self.iteration = self.iteration + 1
        if self.status == AfumigatusCell.Status.RESTING_CONIDIA and \
                self.iteration >= min_iter_to_status_change and \
                random() < probability_status_change:
            self.status = AfumigatusCell.Status.SWELLING_CONIDIA
            self.iteration = 0
        elif self.status == AfumigatusCell.Status.SWELLING_CONIDIA and \
                self.iteration >= min_iter_to_status_change and \
                random() < probability_status_change:
            self.status = AfumigatusCell.Status.HYPHAE
            self.iteration = 0
        elif self.status == AfumigatusCell.Status.DYING:
            self.status = AfumigatusCell.Status.DEAD

    def is_dead(self):
        return self.status == AfumigatusCell.Status.DEAD

    def leave(self, qtty):
        return False

    def move(self, old_voxel, new_voxel):
        pass

    def process_boolean_network(self):
        pass

    @staticmethod
    def rotatation_matrix(phi):
        cos_theta = math.cos(math.pi/4.0)
        sin_theta = math.sin(math.pi / 4.0)

        return np.array([
            [cos_theta * math.cos(phi), -cos_theta*math.sin(phi), sin_theta],
            [math.sin(phi), math.cos(phi), 0.0],
            [-sin_theta*math.cos(phi), sin_theta*math.sin(phi), cos_theta]])
