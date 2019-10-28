from abc import ABC, abstractmethod

class Cell(ABC):
    
    @abstractmethod
    def process_boolean_network(self):
        pass

    @abstractmethod
    def update_status(self):
        pass

    @abstractmethod
    def is_dead(self):
        pass

    @abstractmethod
    def move(self, oldVoxel, newVoxel):
        pass

    @abstractmethod
    def leave(self, qtty):
        pass