"""
This is a script which simulates the RF remapping 
with a memeristive neuron.
"""


import numpy as np 
import matplotlib.pyplot as plt


### Load the dataset.
with open("data.txt","rb") as f:
    aa = f.readlines()

bb = []
for ai in aa:
    ai = ai.decode('utf-8')
    ri = ai.split()
    ri = [float(ii) for ii in ri]
    bb.append(ri)

data1 = np.array(bb)


def act(d,m):
    ## Here, we set network activation function with a threshold for convenience.
    ## which is equivalent to  adding that on the network dynamics.
    R_max = 9.0
    d50 = 0.35

    k0 = 1.11
    k1 = 0.45

    k2 = 30
    k3 = 100
    
    x = (k0*m**4+k1)*d
    n = k2 + k3*m
    y = R_max*((x**n)/(d50**n+x**n))
    out = np.zeros_like(d)
    thr = 0.41
    out[d>=thr]=y[d>=thr]
    return out



### neural activation function
def act_original(d,m):
    R_max = 9.0
    d50 = 0.35

    k0 = 1.11
    k1 = 0.45

    k2 = 30
    k3 = 100
    
    x = (k0*m**4+k1)*d
    n = k2 + k3*m
    y = R_max*((x**n)/(d50**n+x**n))
    out = y
    return out

def generate_w(N):
    # generate network connection, one-dimensional neural network connected 
    # in an uni-direction.
    W = np.zeros((N,N))
    for i in range(N):
        if i+1<N:
            W[i,i+1] = 0.095
        else:
            continue
    return W

def iter_net(x_pre,r_pre,m,W,I_ext):
    # network dynamics iter one step.
    tau = 10
    alpha_tau = 1./tau 
    x_post = x_pre + alpha_tau*(-x_pre+np.dot(r_pre[None,:],W).reshape(-1))+I_ext 

    r_post = act(x_post,m)
    return x_post,r_post

def network(m):
    # 
    N=200
    T = 1200

    I_ext = np.zeros((T,N))
    I_ext[:300,20:21] = 0.09
    W = generate_w(N)

    r_pre = np.zeros((N))
    x_pre = np.zeros((N))

    r_list = []
    x_list = []
    for t in range(T):
        x_post,r_post = iter_net(x_pre,r_pre,m[t],W,I_ext[t])
        r_list.append(r_post.copy())
        x_list.append(x_post.copy())
        x_pre = x_post
        r_pre = r_post 

    r_list = np.array(r_list)
    x_list = np.array(x_list)
    return r_list

def plot_figure(data1,m):
    inp = data1[:,0]
    cs = ['r','b','k','g','c']
    fit_inp =np.arange(0.,1.,0.01)
    mm = np.array([0,0.4,0.5,0.6,0.7])
    for i in range(len(mm)):
        if mm[i] in m:
            fit_y = act_original(fit_inp,mm[i])
            plt.plot(inp,data1[:,i+1],cs[i]+"^")
            plt.plot(fit_inp,fit_y,cs[i],label="m={} V".format(mm[i]))
            plt.xlabel("Input Strength")
            plt.ylabel("Activity")
            plt.legend()

plt.figure(figsize=(4,3))
for m in [0,0.4,0.5,0.6,0.7]:
    plot_figure(data1,[m])

plt.xlim([0.4,1.0])
plt.tight_layout()


plt.savefig("fitting_res.png")
plt.savefig("fitting_res.eps")
    

plt.figure(figsize=(4,3))
m1 = 0.0
m2 = 0.8

plt.subplot(1,1,1)
m = np.zeros((1200))
m[300:1000]=m2
r_list = network(m)
plt.imshow(r_list.T,aspect="auto")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron Index")
plt.colorbar()


plt.tight_layout()
plt.savefig("test1.png")
plt.savefig("test1.eps")
plt.show()





