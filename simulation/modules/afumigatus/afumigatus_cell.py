from simulation.cell_lib.cell import Cell
from random import random, uniform
from enum import IntEnum
import numpy as np
import math

class AfumigatusCell(Cell):
    name = "AfumigatusCell"
    InitAfumigatusBooleanState = [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    #status:
    class Status(IntEnum):
        RESTING_CONIDIA = 0
        SWELLING_CONIDIA = 1
        HYPHAE = 2
        DYING = 3
        DEAD = 4

    #state:
    class State(IntEnum):
        FREE = 0
        INTERNALIZING = 1
        RELEASING = 2

    def __init__(self, x=0, y=0, z=0, ironPool = 0, status = Status.RESTING_CONIDIA, state = State.FREE, isRoot = True, id_in = -1):
        self.id = id_in
        self.iron_pool = ironPool
        self.state = state
        self.status = status
        self.is_root = isRoot
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
        s =  "AfumigatusCell_" + str(self.id) + \
            ' x=' + '%.5f' % self.x + \
            ' y=' + '%.5f' % self.y + \
            ' z=' + '%.5f' % self.z + \
            ' dx=' + '%.5f' % self.dx + \
            ' dy=' + '%.5f' % self.dy + \
            ' dz=' + '%.5f' % self.dz

        if(self.status == Status.RESTING_CONIDIA):
            s = s + ' RESTING_CONIDIA'
        if(self.status == Status.SWELLING_CONIDIA):
            s = s + ' SWELLING_CONIDIA'
        if(self.status == Status.HYPHAE):
            s = s + ' HYPHAE'
        if(self.status == Status.DYING):
            s = s + ' DYING'
        if(self.status == Status.DEAD):
            s = s + ' DEAD'
        if(self.state == State.FREE):
            s = s + ' FREE'
        if(self.state == State.INTERNALIZING):
            s = s + ' INTERNALIZING'
        if(self.state == State.RELEASING):
            s = s + ' RELEASING'
       
        if(self.previous_septa):
            s = s + ' prev=' + str(self.previous_septa.id)
        if(len(self.branch_children) > 0):
            s  = s + ' b_children= ' + str(self.branch_children[0].id)
            for c in self.branch_children[1:]:
                s = s + ', ' + str(c.id)
        if(len(self.elongate_children) > 0):
            s  = s + ' e_children= ' + str(self.elongate_children[0].id)
            for c in self.elongate_children[1:]:
                s = s + ', ' + str(c.id)
        return s

    def set_growth_vector(self, growth_vector):
        self.dx = growth_vector[0]
        self.dy = growth_vector[1]
        self.dz = growth_vector[2]

    def elongate(self, xbin, ybin, zbin): # TODO incorporate geometry growth restriction
        if self.growable and self.status == Status.HYPHAE:# TODO add iron dependence and self.boolean_network[AfumigatusCell.LIP] == 1:
            if(self.x + self.dx < 0 or self.y + self.dy < 0 or self.z + self.dz < 0 \
                or self.x + self.dx > xbin or self.y + self.dy > ybin or self.z + self.dz > zbin):
                return None
            else:
                self.growable = False
                self.branchable = True
                self.iron_pool = self.iron_pool / 2.0;
                child = AfumigatusCell(x=self.x + self.dx, y=self.y + self.dy, z=self.z + self.dz,\
                                             ironPool=0, status=Status.HYPHAE, state=self.state, isRoot=False)
                child.previous_septa = self
                child.iron_pool = self.iron_pool
                child.is_root = False
                self.elongate_children = np.append(self.elongate_children, child)
                return child
        return None

    def branch(self, branch_probability, xbin, ybin, zbin): # TODO incorporate geometry growth restriction
        if self.branchable and self.status == AfumigatusCell.HYPHAE: # TODO add iron dependence  and self.boolean_network[AfumigatusCell.LIP] == 1:
            if random() < branch_probability:
                self.iron_pool = self.iron_pool / 2.0
                growth_vector = [self.dx, self.dy, self.dz]
                B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], growth_vector])
                B_inv = np.linalg.inv(B)
                R = rotatation_matrix(2*random()*math.pi)
                R = np.dot(B, np.dot(R, B_inv))
                growth_vector = np.dot(R, growth_vector)
                
                nextX = growth_vector[0] + self.x
                nextY = growth_vector[1] + self.y
                nextZ = growth_vector[2] + self.z

                if(nextX < 0 or nextY < 0 or nextZ < 0 \
                or nextX > xbin or nextY > ybin or nextZ > zbin):
                    return None
                else:
                    child = AfumigatusCell(x=nextX, y=nextY, z=nextZ,\
                                                ironPool=0, status=Status.HYPHAE, \
                                                state=self.state, isRoot=False)
                    child.set_growth_vector(growth_vector)
                    child.iron_pool = self.iron_pool
                    child.previous_septa = self
                    child.is_root = False
                    self.branch_children = np.append(self.branch_children, child)
                    
                    #self.branchable = False    
                    ##set neighbors to be unbranchable
                    #self.next_branch.branchable = False
                    #if(self.previous_septa):
                    #    self.previous_septa.branchable = False

                    return child
                #self.branchable = False
        return None

    def update_status(self, probability_status_change, min_iter_to_status_change):
        self.iteration = self.iteration + 1
        if self.status == Status.RESTING_CONIDIA and \
                self.iteration >= min_iter_to_status_change and \
                random() < probability_status_change:
            self.status = Status.SWELLING_CONIDIA
            self.iteration = 0
        elif self.status == Status.SWELLING_CONIDIA and \
                self.iteration >= min_iter_to_status_change and \
                random() < probability_status_change:
            self.status = Status.HYPHAE
            self.iteration = 0
        elif self.status == Status.DYING:
            self.status = Status.DEAD

    def is_dead(self):
        return self.status == Status.DEAD

    def leave(self, qtty):
        return False

    def move(self, oldVoxel, newVoxel):
        pass

    def process_boolean_network(self):
        pass

    
    @staticmethod
    def rotatation_matrix(phi):
        cosTheta = math.cos(math.pi/4.0)
        sinTheta = math.sin(math.pi / 4.0)

        return np.array([[cosTheta * math.cos(phi), \
            -cosTheta*math.sin(phi), sinTheta], \
            [math.sin(phi), math.cos(phi), 0.0],\
            [-sinTheta*math.cos(phi), sinTheta*math.sin(phi), cosTheta]])

