"""
This is a script for fitting the data of a memristive neuron.

Since the data of memristor neuron is highly nonlinear,it is difficult 
to fit total parameters in and end-to-end manner.
we adopt a hierachical fitting method, which is a combination techniche 
of grid search and least square fitting method.

We assume the n is dependent on the modulation strength n(m), since other 
parameters are difficult to capture the nonlinear changes of the gain in the 
data.

As the parameter n can only affect the gain of the activation function not 
the x-shift, we simplify the form of n 
is linearly dependent on m and pre-define the equation form of n(m).

The first fitting step:
Grid search d50 and the form of n(m), finding the estimate of parameters of a.
Note the a is dependent on the parameters d and m, a(d,m).

The second fitting step:
A least square fitting method is adopt to fitting the parameters of a(d,m).
We assume the a(d,m) is polynomial form.
"""
import numpy as np 
import matplotlib.pyplot as plt
import pdb
import os
from functools import partial
from scipy.optimize import curve_fit


##### laod dataset.
with open("data.txt","rb") as f:
    aa = f.readlines()

bb = []
for ai in aa:
    ai = ai.decode('utf-8')
    ri = ai.split()
    ri = [float(ii) for ii in ri]
    bb.append(ri)

data1 = np.array(bb)


###### the external form of the activation function.
def act_fun(d,a,d50,n):
    R_max=9.0
    x = d+a*d
    return R_max*((x**n)/(d50**n+x**n))


##### plot the fitting results.
def plot_figure(data1,m,a,d50,n):
    inp = data1[:,0]
    cs = ['r','b','k','g','c']
    fit_inp =np.arange(0.,1.,0.01)
    mm = np.array([0,0.4,0.5,0.6,0.7])
    # plt.figure()
    for i in range(len(mm)):
        if mm[i] in m:
            fit_y = act_fun(fit_inp,a,d50,n)
            plt.plot(inp,data1[:,i+1],cs[i]+"^")
            plt.plot(fit_inp,fit_y,cs[i],label="m={}".format(mm[i]))
            # plt.plot(fit_inp,fit_y,label="m=%0.1f,n=%0.1f,a=%0.1f,b=%0.1f,d50=%0.1f"%(m[0],n,a,b,d50))
            # plt.legend()
            plt.xlabel("Input Strength")
            plt.ylabel("Activity")

## grid search method.
def grid_search(act_fun1,fit_x,fit_y):
    search_space=np.arange(-1,0,0.01)
    best_a = search_space[0]
    best_err = 1000000
    for i in range(len(search_space)):
        pred = act_fun1(fit_x,search_space[i])
        err = np.mean((pred-fit_y)**2)
        if err < best_err:
            best_a = search_space[i]
            best_err = err
    
    print("best_a is %0.2f, err is %0.4f"%(best_a,best_err))
    return best_a


## data under differnt modulation.
mm = [0.,0.4,0.5,0.6,0.7]

################### Parameters which need Manual adjustments ############
## we pre-define the n, the linear form of n can be changed.
## here is a sample. 
n_list=[20+i*15 for i in mm]

### mn define the internal form of the activation function.
global mn
mn=5


path=("figs_test_x{}_{}".format(mn,n_list))
os.makedirs(path,exist_ok=True)

# We search different d50, and get parameter "a" estimation of a under different 
## parameter "d50"
for d50 in np.arange(0.2,0.8,0.05):
    a_fit1 = []
    for i in range(5):
        m = [mm[i]]
        act_fun1 = partial(act_fun,d50=d50,n=n_list[i])
        # popt, pcov = curve_fit(act_fun1, data1[:,0], data1[:,i+1])
        # a = popt.tolist()
        a = grid_search(act_fun1,data1[:,0], data1[:,i+1])
        a_fit1.append(a)
   

    mlist = [0.,0.4,0.5,0.6,0.7]
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    for i in range(5):
        plot_figure(data1,[mlist[i]],a_fit1[i],d50,n_list[i])
    plt.xlim([0.4,1.0])
    plt.title("First Step")


    def fun(m,k1,k2,k3):
        # return k1*m**mn+k2 +k3*0
        return k1*(m)**mn + k2*m

    popt, pcov = curve_fit(fun, np.array(mlist), np.array(a_fit1))
    k1,k2,k3 = popt.tolist()

    xx = np.arange(0.,0.9, 0.01)
    plt.subplot(1,3,2)
    plt.plot(mlist,a_fit1,"*")
    plt.plot(xx,fun(xx,k1,k2,k3))
    plt.title("Second Step")

    plt.subplot(1,3,3)
    for i in range(5): 
        aa = fun(mlist[i],k1,k2,k3)
        plot_figure(data1,[mlist[i]],aa,d50,n_list[i])
    
    plt.xlim([0.4,1.0])
    plt.title("Fitting Results")
    plt.tight_layout()
    
    plt.savefig(path+"/d50_%0.2f_k1_%0.2f_k0_%0.2f_img.png"%(d50,k1,k2))
    



