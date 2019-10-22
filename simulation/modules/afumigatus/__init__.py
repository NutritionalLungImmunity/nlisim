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
        'inhaling': True,
        'trees' : [],
        'isNew' : {},
        'min_iter_to_status_change' : 5,
        'pr_status_change' : 0.05,
        'lastId' : -1
    }
    
    StateClass = AfumigatusState

    def initialize(self, state: State):
        afumigatus: AfumigatusState = state.afumigatus

        c = self.config

        afumigatus.inhaling = True
        afumigatus.trees = []
        afumigatus.isNew = {}
        afumigatus.min_iter_to_status_change = c.getint('min_iter_to_status_change')
        afumigatus.pr_status_change = c.getfloat('pr_status_change')
        afumigatus.lastId = 0 
        afumigatus.sporesLodgeProb = np.ndarray # TODO initialize with something   

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        afumigatus: AfumigatusState = state.afumigatus
        newAf = list()
        print("aspergillus module advancing...")
        # load afumigatus data from state

		# getLastId() = from state 
		
		#update and grow
		#newAfumigatus = list()
        #
		##iterate over roots and then sub trees
		#for(Hyphae<Afumigatus> root: trees):
		#	ArrayList<Hyphae<Afumigatus>> nodesToProcess = new ArrayList<Hyphae<Afumigatus>>()
		#	nodesToProcess.add(root)
		#	
		#	while(!nodesToProcess.isEmpty()):
		#		#get first id
		#		Hyphae<Afumigatus> currHyphae = nodesToProcess.get(0)
		#		ArrayList<Hyphae<Afumigatus>> children = currHyphae.getChildren()
		#		nodesToProcess.addAll(children) # depth first search
		#		
		#		Afumigatus afumigatus = currHyphae.getData()
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
		#diffuseIron()	
		#isNew.clear()
	

        return state
