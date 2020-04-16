# -*- coding: utf-8 -*-
"""
@author: ppauli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import solve_SDP
import time
import matlab.engine
from scipy.io import savemat
import os
from datetime import datetime

INPUT_SIZE = 1
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(INPUT_SIZE,HIDDEN_SIZE)
        self.lin2 = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE)
    
    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        #x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
    def l2_reg(self,lmbd):
        reg_loss = None
        for param in self.parameters():
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param**2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2)**2
        return lmbd * reg_loss
    
    def Lip_reg(self,rho,parameters):
        Lip_loss = None
        i = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                W_bar = torch.tensor(parameters['W{:d}_bar'.format(i)])
                Y = torch.tensor(parameters['Y{:d}'.format(i)])
                if Lip_loss is None:
                    Lip_loss =  rho/2 * torch.sum((param-W_bar)**2) + torch.trace(torch.matmul(Y.t(),(param-W_bar)))
                else:
                    Lip_loss = Lip_loss +  rho/2 * torch.sum((param-W_bar)**2) + torch.trace(torch.matmul(Y.t(),(param-W_bar)))
                i += 1
        return Lip_loss

    def evaluate_MSELoss(self, input, target):
        return 1/len(target)*torch.sum((self(input)-target)**2)

    def extract_weights(self):
        weights = []
        biases = []
        for param_tensor in self.state_dict():
            tensor = self.state_dict()[param_tensor].detach().numpy().astype(np.float64)

            if 'weight' in param_tensor:
                weights.append(tensor)
            if 'bias' in param_tensor:
                biases.append(tensor)
        return weights, biases

    
    def train(self, epochs, lmbd=None, rho=None, parameters=None):
        L_course=[]
        MSE_course=[]
        for i in range(epochs):
            out = self(input)
            
            criterion = nn.MSELoss()    
            loss = criterion(out, target)
            
            if lmbd is not None:
                loss += self.l2_reg(lmbd)

            if rho is not None:
                loss += self.Lip_reg(rho,parameters)
            
            if np.mod(i,1000) == 0:
                MSE = self.evaluate_MSELoss(input,target)
                weights, biases = self.extract_weights()
                Lip = solve_SDP.build_T(weights,biases,net_dims)
                L = Lip["Lipschitz"]
                L_W = np.linalg.norm(weights[0],2)*np.linalg.norm(weights[1],2)
                L_course.append(L)
                MSE_course.append(MSE.item())
                print('Train Epoch: {}; Loss: {:.6f}; MSELoss: {:.6f}; Lipschitz: {:.3f}; ; Trivial Lipschitz: {:.3f}'.format(
                    i, loss.item(), MSE.item(), L, L_W))
                print(Lip["ok"])
            self.zero_grad()
            loss.backward()
            
            optimizer = optim.SGD(self.parameters(), lr=lr)
            # optimizer = optim.Adagrad(self.parameters(), lr=lr)
            # optimizer = optim.Adam(self.parameters(), lr=lr)
            optimizer.step()
        return L_course, MSE_course
            
    def train_Lipschitz(self, parameters,T):
        L_course_Lip = []
        MSE_course_Lip = []
        t=time.time()
        print("Beginnning Lipschitz training")
        for i in range(it_ADMM):
            print("ADMM Iteration # {:d}".format(i))
            L_course, MSE_course = self.train(epochs=epochs, rho=rho, parameters=parameters) #loss update step
            L_course_Lip.append(L_course)
            MSE_course_Lip.append(MSE_course)
            weights, biases=self.extract_weights()
            for i in range(len(weights)):
                parameters.update({
                    'W{:d}'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
                    })    
            parameters = solve_SDP.solve_SDP(parameters,T,net_dims,rho,mu,ind_Lip,L_des) # Lipschitz and Y update steps
                  
        elapsed = time.time() - t
        print("Training Complete after {} seconds".format(elapsed))
        return parameters, L_course_Lip, MSE_course_Lip
                 
if __name__ == '__main__':
    # data
    net_dims = [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]
    torch.manual_seed(2)
    input = Variable(torch.linspace(0,10).view(100,1))
    target = Variable(torch.zeros(input.size()))
    target[10:15]+=1
    target[70:80]+=1
    
    # hyperparameters
    lr = 0.01
    epochs = 10000
    epochs_nom = 100*epochs
    epochs_L2 = 100*epochs
    rho = 2 # ADMM penalty parameter
    mu = 0.0008 # Lip penaly parameter 
    lmbd = 0.0005 # L2 penalty parameter
    it_ADMM = 100
    ind_Lip = 1 # 1 Lipschitz regularization, 2 Enforcing Lipschitz bounds

    # nominal NN
    net = MeinNetz()

    t=time.time()
    print("Beginnning nominal NN training")
    L_course, MSE_course = net.train(epochs=epochs_nom)
    elapsed = time.time() - t
    print("Training Complete after {} seconds".format(elapsed))

    weights, biases = net.extract_weights()
    Lip = solve_SDP.build_T(weights,biases,net_dims)

    # NN with L2 regularizer   
    net_L2 = MeinNetz()

    t=time.time()    
    print("Beginnning L2 training")
    L_course_L2, MSE_course_L2 = net_L2.train(epochs=epochs_L2, lmbd=lmbd)
    elapsed = time.time() - t    
    print("Training Complete after {} seconds".format(elapsed))

    weights_L2, biases_L2 = net_L2.extract_weights()
    Lip_L2 = solve_SDP.build_T(weights_L2,biases_L2,net_dims)
    
    # NN with Lipschitz regularizer   
    net_Lip = MeinNetz()
    net_Lip.load_state_dict(net_L2.state_dict())    

    L_des=Lip_L2["Lipschitz"]  
    parameters = solve_SDP.initialize_parameters(weights,biases)
    parameters_L2 = solve_SDP.initialize_parameters(weights_L2,biases_L2)
    init=1 # 1 initialize from L2-NN, 2 initialize from nominal NN
    if init==1:
        parameters_Lip, L_course_Lip, MSE_course_Lip = net_Lip.train_Lipschitz(parameters=parameters_L2,T=Lip_L2["T"])
    else:
        parameters_Lip, L_course_Lip, MSE_course_Lip = net_Lip.train_Lipschitz(parameters=parameters,T=Lip["T"])       
    
    net_Lip2 = type(net_Lip)()
    net_Lip2.load_state_dict(net_Lip.state_dict())
    with torch.no_grad():
        net_Lip2.lin1.weight = torch.nn.Parameter(torch.tensor(parameters_Lip['W0_bar']))
        net_Lip2.lin2.weight = torch.nn.Parameter(torch.tensor(parameters_Lip['W1_bar']))
    
    weights_Lip, biases_Lip = net_Lip.extract_weights()
    weights_Lip2, biases_Lip2 = net_Lip2.extract_weights()    
    
    Lip_Lip = solve_SDP.build_T(weights_Lip, biases_Lip, net_dims)    
    Lip_Lip2 = solve_SDP.build_T(weights_Lip2, biases_Lip2, net_dims)
       
    # Predictions          
    out=net(input)
    out_L2=net_L2(input)
    out_Lip=net_Lip(input)
    out_Lip2=net_Lip2(input)
    
    #Plots
    now=datetime.now()
    date=now.strftime("%Y-%m-%d_%H-%M-%S")
    
    plt.plot(input, target)
    plt.plot(input, out.detach().numpy())
    plt.plot(input, out_L2.detach().numpy())
    plt.plot(input, out_Lip.detach().numpy())
    plt.plot(input, out_Lip2.detach().numpy())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Targets','Nom','L2','Lip','Lip2'])
    plt.savefig('Results/'+date+'.png')
    plt.show()
        
    plt.plot(MSE_course)
    plt.plot(MSE_course_L2)
    plt.plot(np.array(MSE_course_Lip).reshape(np.array(MSE_course_Lip).size,1))
    plt.xlabel('# 10^3 iterations')    
    plt.ylabel('MSE Loss')
    plt.savefig('Results/'+date+'_1.png')
    plt.show()

    plt.plot(L_course)
    plt.plot(L_course_L2)
    plt.plot(np.array(L_course_Lip).reshape(np.array(L_course_Lip).size,1))
    plt.xlabel('# 10^3 iterations')    
    plt.ylabel('Lipschitz bound')
    plt.savefig('Results/'+date+'_2.png')
    plt.show()
    
    
    # Save Hyperparameters
    hyper = {
        'lr': lr,
        'n_x': net_dims[0],
        'n_h': net_dims[1],
        'n_y': net_dims[2],
        'rho': rho,
        'mu': mu,
        'lambda': lmbd,
        'epochs':  epochs,
        'epochs_nom': epochs_nom,
        'epochs_L2': epochs_L2,
        'iterations_ADMM': it_ADMM,
        'ind_Lip': ind_Lip,
        'Lip_des': L_des        
    }
    
    # Save data and results    
    data={
        'input': np.array(input, dtype=np.float64),
        'target': np.array(target, dtype=np.float64),
        'out': np.array(out.detach().numpy(), dtype=np.float64), 
        'out_L2': np.array(out_L2.detach().numpy(), dtype=np.float64),
        'out_Lip': np.array(out_Lip.detach().numpy(), dtype=np.float64),
        'out_Lip2': np.array(out_Lip2.detach().numpy(), dtype=np.float64),
        'Lip': Lip["Lipschitz"],
        'Lip_L2': Lip_L2["Lipschitz"],
        'Lip_Lip': Lip_Lip["Lipschitz"],
        'Lip_Lip2': Lip_Lip2["Lipschitz"],
        'MSELoss': net.evaluate_MSELoss(input,target).detach().numpy(),
        'MSELoss_L2': net_L2.evaluate_MSELoss(input,target).detach().numpy(),
        'MSELoss_Lip': net_Lip.evaluate_MSELoss(input,target).detach().numpy(),
        'MSELoss_Lip2': net_Lip2.evaluate_MSELoss(input,target).detach().numpy(),
        #'MSELoss_test': net.evaluate_MSELoss(input,target_test).detach().numpy(),
        #'MSELoss_test_L2': net_L2.evaluate_MSELoss(input,target_test).detach().numpy(),
        #'MSELoss_test_Lip': net_Lip.evaluate_MSELoss(input,target_test).detach().numpy(),
        #'MSELoss_test_Lip2': net_Lip2.evaluate_MSELoss(input,target_test).detach().numpy(), 
        'L_course': np.array(L_course),
        'L_course_L2': np.array(L_course_L2),
        'L_course_Lip': np.array(np.array(L_course_Lip).reshape(np.array(L_course_Lip).size,1)),
        'MSE_course': np.array(MSE_course),
        'MSE_course_L2': np.array(MSE_course_L2),
        'MSE_course_Lip': np.array(np.array(MSE_course_Lip).reshape(np.array(MSE_course_Lip).size,1))       
        }
    
    res={}
    res["hyper"]=hyper
    res["data"]=data
    res["weights"]=weights
    res["weights_L2"]=weights_L2
    res["weights_Lip"]=weights_Lip
    res["weights_Lip2"]=weights_Lip2
    res["biases"]=biases
    res["biases_L2"]=biases_L2
    res["biases_Lip"]=biases_Lip
    res["biases_Lip2"]=biases_Lip2
    
    fname = os.path.join(os.getcwd(), 'Results/res_'+date+'.mat')
    data = {'res': np.array(res)}
    savemat(fname, data)
