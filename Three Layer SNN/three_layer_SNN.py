import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import neuron
import linear

# nearly all paramters in this simulation, but it seems no use to store them in a dict...
param = {'G_min': -1.0, # the minimum (maximum) value of conductance, I found if I use real conductance value,  
        'G_max': 1.0,   # which is on 1e-4 level, the gradience will be too small even in two layer net.
        'Rd': 5.0e3,    # this device resistance is mannually set for smaller leaky current?
        'Cm': 3.0e-6,   # real capacitance is absolutely larger than this value
        'Rs': 1.0,      # this series resistance value is mannually set for larger inject current?
        'Vth': 0.8,     # this is the real device threshould voltage
        'V_reset': 0.0, 
        'dt': 1.0e-6,   # every time step is dt, in the one-order differential equation of neuron
        'T_sim': 10,   # could control total spike number collected
        'dim_in': 784,
        'dim_h': 100,
        'dim_out': 10,
        'amp' : 4.0,    # the gain of TIAs
        'q_bit': 7,     # quantize bit 
        'epoch': 10,
        'batch_size': 128,
        'learning_rate': 0.3, # I am not sure if this lr is too large
        'data_dir': './MNIST',
        'train_file': 'trainning_log_7bit.txt',
        'test_file': 'test_log.txt',
        'model_dir': 'Model.pth'
}

def Poisson_encoder(x):
    '''
    To encode the image pixels to poisson event.

    input: a batch of input data x.
    output: a batch of poisson encoded 1.0 or 0.0 with the same shape as x,
            the possibility of a pixle to be encoded as 1.0 is propotional to the pixel value.
    '''
    out_spike = torch.rand_like(x).le(x).float()
    return out_spike

class Three_Layer_SNN(nn.Module):
    '''
    This net model contains 2 linear layer, 2 self-defined BatchNorm layer and 2 Neuron layer.
    
    linear layer: a memristor crossbar on which the MAC operation is implemented.
    BatchNorm layer: a row of TIA as the output interface of the pre-linear layer, normalize the
                    output current to  -2.0~2.0 V voltage.
    neuron layer: nonliear activation, receive input voltage and output spikes, spiking rate is taken
                    in loss computing.
    '''
    def __init__(self, param):
        super().__init__()
        self.linear1 = linear.MAC_Crossbar(param['dim_in'], param['dim_h'], 
                                            param['G_min'], param['G_max'], param['q_bit'])
        self.BatchNorm1 = linear.TIA_Norm(784.0, 0.0, 200.0)    # the paramters of TIA are mannually set for moderate input voltage to neurons
        self.neuron1 = neuron.LIFNeuron(param['batch_size'], param['dim_h'], param['Rd'], param['Cm'],
                                            param['Rs'], param['Vth'], param['V_reset'], param['dt'])
        self.linear2 = linear.MAC_Crossbar(param['dim_h'], param['dim_out'], 
                                            param['G_min'], param['G_max'], param['q_bit'])
        self.BatchNorm2 = linear.TIA_Norm(100.0, 0.0, 200.0)    # same as above
        self.neuron2 = neuron.LIFNeuron(param['batch_size'], param['dim_out'], param['Rd'], param['Cm'], 
                                            param['Rs'], param['Vth'], param['V_reset'], param['dt'])

    def forward(self, input_vector):
        out_vector = self.linear1(input_vector)
        # debug print, very useful to see what happend in every layer
        #print('0', out_vector.max())
        out_vector = self.BatchNorm1(out_vector)
        #print('1', out_vector.max())
        out_vector = self.neuron1(out_vector)
        #print('2', out_vector.sum(1).max())
        out_vector = self.linear2(out_vector)
        #print('3', out_vector.max())
        out_vector = self.BatchNorm2(out_vector)
        #print('4', out_vector.max())
        out_vector = self.neuron2(out_vector)
        #print('5', out_vector.sum(1).max())
        return out_vector

    def reset_(self):
        '''
        Reset all neurons after one forward pass,
        to ensure the independency of every input image.
        '''
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()

    def quant_(self):
        '''
        The quantization function in pytorch only support int8,
        so we need our own quant function for adjustable quantization precision.
        '''
        for item in self.modules():
            if hasattr(item, 'Gquant_'):
                #debug print：
                #print(item.weight.max())
                item.Gquant_()
                #debug print：
                #print(item.weight.max())

trainset = torchvision.datasets.MNIST(root=param['data_dir'], train=True,
                                        download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root=param['data_dir'], train=False,
                                        download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=param['batch_size'],
                                            shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=param['batch_size'],
                                            shuffle=False, drop_last=True)


#Train the SNN with BP
net = Three_Layer_SNN(param)
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), param['learning_rate'])
# I have tried Adam and SGD, but only Adam performs well with my Gquant_ function
# Before you use Gquant_, you should debug to figure out a proper lr, because too small lr 
# will induce too small delta w (smaller than the quantization error), then your net parameter 
# will get no change.

for epoch in range(param['epoch']):
    net.train()
    train_accuracy = []
    for img, label in trainloader:
        img = img.reshape(-1, 28*28)
        spike_num_img = torch.zeros(param['batch_size'], param['dim_out'])
        for t in range(param['T_sim']):
            img = Poisson_encoder(img)
            #debug print：
            #print(img.sum())
            out_spike = net(img)
            spike_num_img += out_spike
        
        spike_rate = spike_num_img/param['T_sim']
        #debug print：
        #print(spike_num_img.max())
        #print('one batch end')
        loss = loss_func(spike_rate, label)

        #net.zero_grad()
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            net.reset_()    # reset the neuron voltage every batch, to ensure independency between batchs
            net.quant_()    # quantize the weights after weight update
        
        train_accuracy.append((spike_rate.max(1)[1] == label).float().mean().item())
    accuracy_epoch = np.mean(train_accuracy)
    print('tranning epoch %d: the SNN loss is %.6f' %(epoch, loss), end=' ') 
    print('trainning accuracy: %.4f' %accuracy_epoch, end=' ')

# validation by testset every epoch to see if the network is overfitted
    net.eval()
    validation_accuracy = []
    with torch.no_grad():
        for img_test, label_test in testloader:
            img_test = img_test.reshape(-1, 28*28)
            spike_num_img_test = torch.zeros(param['batch_size'], param['dim_out'])
            for t in range(param['T_sim']):
                img_test = Poisson_encoder(img_test)
                out_spike = net(img_test)
                spike_num_img_test += out_spike
            validation_accuracy.append((spike_num_img_test.max(1)[1]==label_test).float().mean().item())
        accuracy_val = np.mean(validation_accuracy)
        print('validation accuracy: %.4f' %accuracy_val)

    with open(param['train_file'], 'a') as f_t:
        s = str(epoch).ljust(6,' ') + str(round(loss.item(), 6)).ljust(12,' ')
        s += str(round(accuracy_epoch, 4)).ljust(10, ' ') + str(round(accuracy_val, 4)).ljust(10, ' ') + '\n'
        f_t.write(s)

torch.save(net.state_dict(), param['model_dir'])

# Test process after training
net_test = Three_Layer_SNN(param)
print('Loading Model, please wait......')
net_test.load_state_dict(torch.load(param['model_dir']))
print('Model loaded successfully!')
list_num_spike = []
for i in range(10):
    list_num_spike.append([0])
    list_num_spike[i].append(torch.zeros(param['dim_out']))

with torch.no_grad():
    for img_test, label_test in testloader:
        img_test = img_test.reshape(-1,28*28)
        spike_num_img_test = torch.zeros(param['batch_size'], param['dim_out'])
        for t in range(param['T_sim']):
            img_test = Poisson_encoder(img_test)
            out_spike = net_test(img_test)
            spike_num_img_test += out_spike
        pred_label = F.one_hot(spike_num_img_test.max(1)[1], num_classes = 10) # convert the max neuron output index to onehot vector 
        for j in range(label_test.size(0)):
            index = label_test[j]
            list_num_spike[index][0] += 1
            #list_num_spike[index][1] += spike_num_img_test[j] # statistics of neuron output spike number per image
            list_num_spike[index][1] += pred_label[j] #statistics of prediction for every input image

#with open('heat_map.txt','a') as f2:
with open('confusion_matrix.txt','a') as f2:
    for i in range(len(list_num_spike)):
        s = str(list_num_spike[i][0]) + ' ' + str(list_num_spike[i][1].numpy()).replace('[','').replace(']','') + '\n'
        f2.write(s)



        
        
    

    








