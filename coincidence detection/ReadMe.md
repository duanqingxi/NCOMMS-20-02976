## Synchronization detection

## System Requirements

+ 64 bit operating systems（Windows or Mac）

+ Anaconda

+ Python 3.7.6 

+ MATLAB 2018b

 ## Installation Guide

Users are supposed to install Anaconda first, then install Brian2 in the Anaconda root environment.

Install Anaconda: https://www.anaconda.com/products/individual.

It takes about thirty minutes.

Then install Brian2 from the conda-forge channel:

conda install -c conda-forge brian2

It takes about two minutes.

Install MATLAB: https://www.mathworks.com/downloads

It takes about two hours.

## Demo

After successfully install Brian2, users can directly paste the "synchronization detection" code on JupyterLab (an application in anaconda) and run.

The expected output includes data and three figures.

A 1000*1 data represents the membrane potential during operation time(100 us). The sampling interval is 0.1us.
Another data recodes the time of each spike firing. 

Figure 1 represents LIF neuron's response curve, whose parameters are extracted from the experiments. 
Figure 2 represents the membrane potential during operation time when input is uncorrelated.
Figure 3 represents the membrane potential during operation time when input is correlated.

Because the input is a Poisson input with randomness, there will be some differences in results from run to run. 

Users can also directly paste the "plot current spikes" code on MATLAB and run. The code recodes an example based on the above process and plots the current response.


The run time is about 1 second.

## Instructions for use

Paste the "synchronization detection" code on JupyterLab (an application in anaconda) and run.
Paste the "plot current spikes" code on MATLAB and run.
Users are suggested to change parameters and see the networks performance.

