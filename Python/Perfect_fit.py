import numpy as np
import matplotlib.pyplot as plt

# input data
n = 3
x = np.random.uniform(low=0.0, high=1.0, size=n)
v = np.random.uniform(low=-0.1, high=0.1, size=n)

# desired output
d = []
for i in range(n):
    d.append(np.sin(20*x[i]) + (3*x[i]) + v[i])


# feed-forward activation functions
def act_fun(v):
    return np.tanh(v)

def act_op(v):
    return v

# feedback activation functions
def derv_act_fun(v):
    return (1 - np.tanh(v)**2)

def derv_act_op(v):
    return 1

# weight initialization
N = 4
w_input = np.random.uniform(low=-5, high=5, size=N)
w_bias = np.random.uniform(low=-1, high=1, size=N)
w_output = np.random.uniform(low=-5, high=5, size=N)
w_final = np.random.uniform(low=-1, high=1, size=1)
eta = 6

list_mse = []
z = 0
counter = 0
while(counter<3):
    # feed-forward network
    u = []
    y = []
    alphas = []
    betas = []
    for i in range(n):
        print("i1",i)
        v = []
        temp = []
        for j in range(N):
            alpha = (x[i]*w_input[j]) + w_bias[j]
            temp.append(alpha)
            v.append(act_fun(alpha))
        print("v",v)    
        print("\n")
        alphas.append(temp)
        print("Alphas",alphas)
        print("\n")
        u.append(v)
        print("List of u",v)
        print("\n")
        beta = np.matmul(np.array(u[i]),w_output) + w_final
        betas.append(beta[0])
        print("Beta",beta)
        print("\n")
        y.append(act_op(beta[0]))
        print("y",y)
        print("\n")
        counter+=1
        
        
        e = -((d[i] - y[i])*eta*2)/n
        for j in range(N):
            print("i",i)
            print("j",j)
            print("The 2D thingy which is u",u[i][j])
            delta_u = e * u[i][j] 
            print("Delta",delta_u)
        
    
