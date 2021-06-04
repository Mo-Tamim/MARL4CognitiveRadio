# SEcondary user  Class
import torch

class SU:

    def __init__(self): 
        self.id = 0 
        self.horizon = 20 
        self.timer = 0 
        self.Warnings = 0 # number or warning recieved from PUs 
    
    def sensing (self): 
        # select / call action function from "Agent class"
        return SensingResults

    
    def detectCollision(self, WAR):         
        if WAR == 999: 
            self.Warnings += 1 
