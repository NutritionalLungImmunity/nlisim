import numpy as np
import math

class Util():
    
    cosTheta = math.cos(math.pi/4.0)
    sinTheta = math.sin(math.pi / 4.0)
    
    @staticmethod
    def rotatation_matrix(phi):
        return np.array([[Util.cosTheta * math.cos(phi), -Util.cosTheta*math.sin(phi), Util.sinTheta],\
                         [math.sin(phi), math.cos(phi), 0.0],\
                         [-Util.sinTheta*math.cos(phi), Util.sinTheta*math.sin(phi), Util.cosTheta]])
