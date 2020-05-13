# Three-layer SNN simulation

## Dataset

MNIST handwritten dataset has been included in ./MNIST folder



## System Requirements

+ Python 3.7.6

+ Pytorch 1.4.0

+ Numpy 1.18.1

  

 ## Installation Guide

Users are supposed to install the libraries mentioned above, then open the repository in any Python interpreter, run the two_layer_SNN.py code directly.

## Demo

The program will first download the MNIST dataset if you do not have the MNIST data in your current work directory.

Then the training process starts, it will print the loss, training accuracy and validation accuracy every epoch, like:

`training epoch 0: the SNN loss is 2.0374 training accuracy: 0.1923 validation accuracy: 0.2058`

Those training results will be written to a training_log.txt file, every row in the .txt file is the training results of one epoch, it seems like:

`0     2.171238    0.2757    0.5247    
1     2.175548    0.5213    0.5742    
2     2.122161    0.5596    0.6931       `    

...

`97    2.070796    0.8075    0.8364    
98    2.072713    0.8217    0.8343    
99    2.069286    0.8145    0.8324`

After training, the test process starts, you will see:

`Loading Model, please wait......`

`Model loaded successfully!`

if you successfully save the trained network and load it again.

Then a heat_map.txt file will generate, which contains the statistics of the network output on testset. 

## Instructions for use

Users can change the simulation parameters by manually set the parameter value in the dict `param` at the beginning of two_layer_SNN.py file. 



## License

[MIT](https://github.com/duanqingxi/NCOMMS-20-02976/blob/master/LICENSE) © Zhaokun Jing



