"""
Afumigatus Module
"""
from math import floor

import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State
from simulation.validation import ValidationError
from simulation.modules.afumigatus.afumigatus_cell import AfumigatusCell

from typing import List
from typing import Dict


@attr.s(kw_only=True, repr=False)
class AfumigatusState(ModuleState):
    #Afumigatus data struct
    #trees: list(AfumigatusCell)# = new ArrayList<Hyphae<Afumigatus>>()
	#isNew: Dict[AfumigatusCell,  bool] # = new HashMap<Integer, Boolean>()    
    sporesLodgeProb: np.ndarray #float[][][]
    inhaling: bool
    lastId:int = attr.ib(default=-1)
    min_iter_to_status_change: int = attr.ib(default=0)
    pr_status_change: float = attr.ib(default=0.5)

    ## connector things from java 
    # Map<Integer, E> newAfumigatus
	# Set<Integer> removeAfumigatus
	
    #def __repr__(self):
    #    return f'AfumigatusState(concentration, wind_x, wind_y, source)'
    pass

class Afumigatus(Module):
    name = 'afumigatus'
    defaults = {
        'inhaling': 'True',
        'trees' : 'np.array([])',
        #'isNew' : '{}',
        'min_iter_to_status_change' : '5',
        'pr_status_change' : '0.05',
        'num_spore' : '0',
        'lastId' : '-1'
    }
    
    StateClass = AfumigatusState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus

        c = self.config

        afumigatus.inhaling = True
        afumigatus.trees = np.array([])
        #afumigatus.isNew = {}
        afumigatus.min_iter_to_status_change = c.getint('min_iter_to_status_change')
        afumigatus.pr_status_change = c.getfloat('pr_status_change')
        afumigatus.branch_probability = c.getfloat('branch_probability')
        afumigatus.init_spores = c.getfloat('init_spores')
        afumigatus.num_spore = 0 
        afumigatus.last_id = -1 
        afumigatus.spores_lodge_prob = np.ndarray # TODO initialize with something 

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        afumigatus: AfumigatusState = state.afumigatus
        
        #print("aspergillus module advancing...")
        # load afumigatus data from state
        num_spore = afumigatus.num_spore
        init_spores = afumigatus.init_spores
        trees = afumigatus.trees
        print ('----------------------------------------------------------------------------------------------------------------------')
        #for a in trees:
        #    print(a)
		# getLastId() = from state 
        #print ('----------------------------------------------------------------------------------------------------------------------')
		#update and grow
        isNew = {}
        print(len(trees))
		##iterate over roots and then sub trees
		#for(Hyphae<Afumigatus> root: trees):
        for root in trees:
            print(root)
            nodesToProcess = np.array([])
            nodesToProcess = np.append(nodesToProcess, root)
        	
            while(len(nodesToProcess) > 0):
                curr_af = nodesToProcess[0]
                nodesToProcess = np.delete(nodesToProcess, 0, axis=0)
                sss = (str(curr_af.id) + ' -> ')
                #print(curr_af)
                elongate_children = curr_af.elongate_children
                branch_children = curr_af.branch_children
               
                if(len(elongate_children) > 0):
                    sss = sss + ('[' + str(elongate_children[0].id))
                    for c in elongate_children[1:]:
                        sss = sss + (', ' + str(c.id))
                    sss = sss + (']')
                    for c in elongate_children:
                        nodesToProcess = np.append(nodesToProcess, c)
                        
                if(len(branch_children) > 0):
                    sss = sss + ('[' + str(branch_children[0].id))
                    for c in branch_children[1:]:
                        sss = sss + (', ' + str(c.id))
                    sss = sss + (']')
                    for c in branch_children:
                        nodesToProcess = np.append(nodesToProcess, c)
                
                print(sss)
                curr_af.update_status(afumigatus.pr_status_change, afumigatus.min_iter_to_status_change)

                #grow, branch
                new_af = curr_af.elongate(20,20,20)
                if(new_af):
                    new_af.id = afumigatus.last_id + 1
                    afumigatus.last_id += 1
                    afumigatus.num_spore += 1
                new_af2 = curr_af.branch(afumigatus.branch_probability, 20,20,20)
                if(new_af2):
                    new_af2.id = afumigatus.last_id + 1
                    afumigatus.last_id += 1
                    afumigatus.num_spore += 1
                if(curr_af.previous_septa == None and not(curr_af.switched) and len(curr_af.elongate_children) > 0):
                    #we are at root, so can grow opposite direction
                    # TODO add growth delay based on random() < prob_dual elongate
                    curr_af.set_dx(curr_af.get_dx() * - 1)
                    curr_af.set_dy(curr_af.get_dy() * - 1)
                    curr_af.set_dz(curr_af.get_dz() * - 1)
                    curr_af.set_growable(True)
                    curr_af.switched = True
                
		#		if(afumigatus.isAlive()) :
		#			if(afumigatus.isLodged()) :
		#				afumigatus.updateBooleanNetwork() #update the boolean network
		#				afumigatus.updateStatus()
		#				if((newAfumigatus = afumigatus.elongate(afumigatusSet, geo))!=null) {
		#					connector.add(newAfumigatus)
		#					
		#					Hyphae<Afumigatus> newHyphae = new Hyphae<Afumigatus>(newAfumigatus)
		#					newHyphae.setParent(currHyphae)
		#					currHyphae.addChild(newHyphae)
		#					
		#					isNew.put(newHyphae.getId(), true)
		#				
		#				if((newAfumigatus = afumigatus.branch(afumigatusSet, geo))!=null):						
		#					Hyphae<Afumigatus> newHyphae = new Hyphae<Afumigatus>(newAfumigatus)
		#					newHyphae.setParent(currHyphae)
		#					currHyphae.addChild(newHyphae)
		#					
		#					isNew.put(newHyphae.getId(), true)
        #
		#				if(currHyphae.getParent() == null):
		#					#we are at root, so can grow opposite direction
		#					afumigatus.setDx(afumigatus.getDx() * - 1)
		#					afumigatus.setDy(afumigatus.getDy() * - 1)
		#					afumigatus.setDz(afumigatus.getDz() * - 1)
		#					afumigatus.setGrowable(true)
		#		 else:
		#			#remove afumigatus if dead
		#			trees.addAll(children) #all children of node become new roots
		#			root.remove(currHyphae)
		#		
		#		nodesToProcess.remove(0)
		#    	
		#    
		#
		##add new spores
		#addNewSpores(afumigatusSet.size())
        #print([num_spore, init_spores])
        if (num_spore < init_spores):
            afumigatus.last_id += 1
            last_id = afumigatus.last_id
            af = AfumigatusCell(x=10, y=10, z=10, ironPool = 0, status = AfumigatusCell.RESTING_CONIDIA, state = AfumigatusCell.FREE, isRoot = True, id_in = last_id)
            afumigatus.trees = np.append(afumigatus.trees, af)
            afumigatus.num_spore += 1
        #diffuseIron()	
		#isNew.clear()
        return state
