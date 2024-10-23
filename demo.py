
import numpy as np

class gates():
    def __init__(self, wNOT = np.array([0]), wAND = np.array([0,0]), wOR = np.array([0,0]), bOR = 0, bAND = 0, bNOT = 0, learning_rate = 0.1):
        self.wNOT = wNOT
        self.wAND = wAND
        self.wOR = wOR
        self.bOR = bOR
        self.bAND = bAND
        self.bNOT = bNOT
        self.learning_rate = learning_rate

    def activation(self,x):
        return 1 if x>=0 else 0
    
    def perceptron(self,x,w,b):
        out = np.dot(x,w) + b
        return self.activation(out)
    
    def NOT_function(self,x):
        x = np.array(x)
        return self.perceptron(x,self.wNOT,self.bNOT)
    
    def AND_function(self,x):
        x = np.array(x)
        return self.perceptron(x,self.wAND,self.bAND)
    
    def OR_function(self,x):
        x = np.array(x)
        return self.perceptron(x,self.wOR,self.bOR)
    
    def XOR_function(self,x):
        x = np.array(x)
        
        y0 = self.NOT_function(x[0])
        y1 = self.NOT_function(x[1])
        z0 = self.AND_function([y0,x[1]])
        z1 = self.AND_function([y1,x[0]])
        
        return self.OR_function([z0,z1])
    
    def NXOR_function(self,x):
        return self.NOT_function(self.XOR_function(x))
    
    def update_weight(self,inp,out,tar,weight):
        res = weight + self.learning_rate*(tar - out)*inp
        return res
    
    def update_bias(self,b,out,tar,):
        res = b + 0.3*(tar - out)
        return res

    def update_not(self):
        inp = np.array([1,0])
        tar = np.array([0,1])
        
        i = 0
        while i <len(inp):
            out = self.NOT_function(inp[i])
            if(tar[i] != out):
                weight = self.wNOT
                self.wNOT = self.update_weight(inp[i],out,tar[i],self.wNOT)
                if (weight == self.wNOT).all():
                    self.bNOT = self.update_bias(self.bNOT,out,tar[i])
                #print(self.wNOT,self.bNOT)
                i = 0
            else:
                i += 1
                
    def update_and(self):
        inp = np.array([(0,0),(1,0),(0,1),(1,1)])
        tar = np.array([0,0,0,1])
        
        i = 0
        while i <len(inp):
            out = self.AND_function(inp[i])
            if(tar[i] != out):
                weight = self.wAND
                self.wAND = self.update_weight(inp[i],out,tar[i],self.wAND)
                if (weight == self.wAND).all():
                    self.bAND = self.update_bias(self.bAND,out,tar[i])
                #print(self.wAND,self.bAND)
                i = 0
            else:
                i += 1
                
    def update_or(self):
        inp = np.array([(0,0),(1,0),(0,1),(1,1)])
        tar = np.array([0,1,1,1])
        
        i = 0
        while i <len(inp):
            out = self.OR_function(inp[i])
            if(tar[i] != out):
                weight = self.wOR
                self.wOR = self.update_weight(inp[i],out,tar[i],self.wOR)
                if (weight == self.wOR).all():
                    self.bOR = self.update_bias(self.bOR,out,tar[i])
                #print(self.wOR,self.bOR)
                i = 0
            else:
                i += 1


a = gates(
    wNOT=1,
    bNOT=1,
    learning_rate=.1
)

a.update_and()
a.update_not()
a.update_or()
print(a.NOT_function(0),a.NOT_function(1))
print(a.XOR_function([0,0]),a.XOR_function([0,1]),a.XOR_function([1,0]),a.XOR_function([1,1]))