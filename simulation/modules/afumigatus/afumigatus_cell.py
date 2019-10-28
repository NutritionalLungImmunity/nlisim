from simulation.cell_lib.cell import Cell
from simulation.cell_lib.util import Util
from random import random, uniform
import numpy as np
import math

class AfumigatusCell(Cell):
    name = "AfumigatusCell"
    InitAfumigatusBooleanState = [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    #status:
    RESTING_CONIDIA = 0
    SWELLING_CONIDIA = 1
    HYPHAE = 2
    DYING = 3
    DEAD = 4

    #state:
    FREE = 0
    INTERNALIZING = 1
    RELEASING = 2

    def __init__(self, x=0, y=0, z=0, ironPool = 0, status = 0, state = 0, isRoot = True, id_in = -1):
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

        self.children = []
        self.previous_septa = None
        self.Fe = False

    def get_x (self):
        return self.x 
    def get_y (self):
        return self.y 
    def get_z (self):
        return self.z 
    def get_dx(self):
        return self.dx
    def get_dy(self):
        return self.dy
    def get_dz(self):
        return self.dz

    def set_growable(self, val):
        self.growable = val
    def set_x (self, val):
        self.x  = val
    def set_y (self, val):
        self.y  = val
    def set_z (self, val):
        self.z  = val
    def set_dx(self, val):
        self.dx = val
    def set_dy(self, val):
        self.dy = val
    def set_dz(self, val):
        self.dz = val

    def __str__(self):
        s =  "AfumigatusCell_" + str(self.id) + \
            ' x=' + '%.5f' % self.x + \
            ' y=' + '%.5f' % self.y + \
            ' z=' + '%.5f' % self.z + \
            ' dx=' + '%.5f' % self.dx + \
            ' dy=' + '%.5f' % self.dy + \
            ' dz=' + '%.5f' % self.dz

        if(self.status == AfumigatusCell.RESTING_CONIDIA):
            s = s + ' RESTING_CONIDIA'
        if(self.status == AfumigatusCell.SWELLING_CONIDIA):
            s = s + ' SWELLING_CONIDIA'
        if(self.status == AfumigatusCell.HYPHAE):
            s = s + ' HYPHAE'
        if(self.status == AfumigatusCell.DYING):
            s = s + ' DYING'
        if(self.status == AfumigatusCell.DEAD):
            s = s + ' DEAD'
        if(self.state == AfumigatusCell.FREE):
            s = s + ' FREE'
        if(self.state == AfumigatusCell.INTERNALIZING):
            s = s + ' INTERNALIZING'
        if(self.state == AfumigatusCell.RELEASING):
            s = s + ' RELEASING'
       
        if(self.previous_septa):
            s = s + ' prev=' + str(self.previous_septa.id)
        if(len(self.children) > 0):
            s  = s + ' children= ' + str(self.children[0].id)
            for c in self.children[1:]:
                s = s + ', ' + str(c.id)
        return s

    def set_growth_vector(self, growth_vector):
        self.dx = growth_vector[0]
        self.dy = growth_vector[1]
        self.dz = growth_vector[2]

    def elongate(self, xbin, ybin, zbin):
        if self.growable and self.status == AfumigatusCell.HYPHAE:# and self.boolean_network[AfumigatusCell.LIP] == 1:
            if(self.x + self.dx < 0 or self.y + self.dy < 0 or self.z + self.dz < 0 \
                or self.x + self.dx > xbin or self.y + self.dy > ybin or self.z + self.dz > zbin):
                return None
            else:
                self.growable = False
                self.branchable = True # TODO on make branchable if previous is not branchable
                self.iron_pool = self.iron_pool / 2.0;
                child = AfumigatusCell(x=self.x + self.dx, y=self.y + self.dy, z=self.z + self.dz,\
                                             ironPool=0, status=AfumigatusCell.HYPHAE, state=self.state, isRoot=False)
                child.previous_septa = self
                child.iron_pool = self.iron_pool
                child.is_root = False
                self.children.append(child)
                return child
        return None

    def branch(self, branch_probability, xbin, ybin, zbin):
        if self.branchable and self.status == AfumigatusCell.HYPHAE: #and self.boolean_network[AfumigatusCell.LIP] == 1:
            if random() < branch_probability:
                self.iron_pool = self.iron_pool / 2.0
                growth_vector = [self.dx, self.dy, self.dz]
                B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], growth_vector])
                B_inv = np.linalg.inv(B)
                R = Util.rotatation_matrix(2*random()*math.pi)
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
                                                ironPool=0, status=AfumigatusCell.HYPHAE, \
                                                state=self.state, isRoot=False)
                    child.set_growth_vector(growth_vector)
                    child.iron_pool = self.iron_pool
                    child.previous_septa = self
                    child.is_root = False
                    self.children.append(child)
                    #set neighbors to be unbranchable
                    self.branchable = False                    
                    #self.next_branch.branchable = False
                    #if(self.previous_septa):
                    #    self.previous_septa.branchable = False

                    return child
                #self.branchable = False
        return None

    def update_status(self, probability_status_change, min_iter_to_status_change):
        self.iteration = self.iteration + 1
        if self.status == AfumigatusCell.RESTING_CONIDIA and \
                self.iteration >= min_iter_to_status_change and \
                random() < probability_status_change:
            self.status = AfumigatusCell.SWELLING_CONIDIA
            self.iteration = 0
        elif self.status == AfumigatusCell.SWELLING_CONIDIA and \
                self.iteration >= min_iter_to_status_change and \
                random() < probability_status_change:
            self.status = AfumigatusCell.HYPHAE
            self.iteration = 0
        elif self.status == AfumigatusCell.DYING:
            self.status = AfumigatusCell.DEAD

        # #is this phagocyte module dependent?
        # if self.state == AfumigatusCell.INTERNALIZING or self.state == AfumigatusCell.RELEASING:
        #    self.state = AfumigatusCell.FREE

    def is_dead(self):
        return self.status == AfumigatusCell.DEAD

    def leave(self, qtty):
        return False

    #def die(self):
    #    AfumigatusCell.total_cells = AfumigatusCell.total_cells - 1

    def move(self, oldVoxel, newVoxel):
        pass

    def process_boolean_network(self):
        pass

    # #is this phagocyte module dependent?
    #def is_internalized(self):
    #    return self.state == AfumigatusCell.INTERNALIZING
    #
    #def is_internalizing(self):
    #    return self.state == AfumigatusCell.INTERNALIZING

    #def diffuse_iron(self, afumigatus=None):
    #    if afumigatus == None:
    #        if self.is_root:
    #            self.diffuse_iron(self)
    #    else:
    #        if afumigatus.next_septa is not None and afumigatus.next_branch is None:
    #            current_iron_pool = afumigatus.iron_pool
    #            next_iron_pool = afumigatus.next_septa.iron_pool
    #            iron_pool = (current_iron_pool + next_iron_pool) / 2.0
    #            afumigatus.iron_pool = iron_pool
    #            afumigatus.next_septa.iron_pool = iron_pool
    #            self.diffuse_iron(afumigatus.next_septa)
    #        elif afumigatus.next_septa is not None and afumigatus.next_branch is not None:
    #            current_iron_pool = afumigatus.iron_pool
    #            next_iron_pool = afumigatus.next_septa.iron_pool
    #            branch_iron_pool = afumigatus.next_branch.iron_pool
    #            iron_pool = (current_iron_pool + next_iron_pool + branch_iron_pool) / 3.0
    #            afumigatus.iron__pool = iron_pool
    #            afumigatus.next_septa.iron_pool = iron_pool
    #            afumigatus.next_branch.iron_pool = iron_pool
    #            self.diffuse_iron(afumigatus.next_branch)
    #            self.diffuse_iron(afumigatus.next_septa)
    #
    #def has_iron(self):
    #    self.Fe = Util.hillProbability(self.iron_pool, Constants.Kma) > random()
    #
    #def inc_iron_pool(self, qtty):
    #    self.iron_pool = self.iron_pool + qtty
    #    AfumigatusCell.total_iron = AfumigatusCell.total_iron + qtty
