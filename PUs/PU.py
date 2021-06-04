# Primary user Class 

import torch 
class PU: 
    def __init__(self,Horizon=20): 
        self.id = 0 
        self.horizon = Horizon 
        self.TxPattern = torch.zeros(self.horizon, )
        self.timer = 0 
    
    def createTxPattern(self):
        self.TxPattern = torch.round(torch.rand((self.horizon,)))

    def issueWarning(self):
        return 999   # this will return a warining to the SU in order to avoid collision 

    def detectCollision(self, NACK):         
        if NACK == 1 : 
            self.issueWarning() 

if  __name__ == "__main__":
    PU_test = PU(2)
    PU_test.createTxPattern()
    print(PU_test.TxPattern)

    print(torch.__version__)
