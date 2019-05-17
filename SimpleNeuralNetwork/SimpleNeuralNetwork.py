#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:42:05 2019

@author: shashanksoni092
"""

#Declaring a class named NeuralNetwork( a Structure)

import numpy as np
import scipy.special

class NeuralNetwork:
    
    #initialise the neuralNetwork parameters

    # number of input, hidden and output nodes
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
        #Set number of nodes in each layer input,hidden,output
        self.inodes=inputNodes
        self.onodes=outputNodes
        self.hnodes=hiddenNodes
        self.lr=learningRate #setting the learning rate
        #so we are ready to create weights b/w (i/p and hidden layer (wih) )
        #and (hidden and o/p (who))
        self.wih=(np.random.rand(self.hnodes,self.inodes)-0.5)
        self.who=(np.random.rand(self.onodes,self.hnodes)-0.5)
        self.activation_function=lambda x:scipy.special.expit(x)#1/1+pow(e,-x) sigmoid fun
        
        pass
    
    #train the neural network
    def train(self,input_list,target_list):
        
        inputs=np.array(input_list,ndmin=2).T
        targets=np.array(target_list,ndmin=2).T
        #calculating i/p signal to hidden layer
        hidden_inputs=np.dot(self.wih,inputs)
        
        hidden_outputs=self.activation_function(hidden_inputs)
        
        #final input
        final_inputs=np.dot(self.who,hidden_outputs)
        
        #sigmoid function
        final_outputs=self.activation_function(final_inputs)
        
        output_errors=targets-final_outputs        
        
        
        hidden_errors=np.dot(self.who.T,output_errors)
        
        self.who +=self.lr*numpy.dot((output_errors*final_outputs*(1-final_outputs))
        ,np.transpose(hidden_outputs))
        
        self.wih +=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs))
        ,np.transpose(inputs))
        
        pass
    
    #query the neural network
    def query(self,input_list):
        inputs=np.array(input_list,ndmin=2).T
        
        #calculating i/p signal to hidden layer
        hidden_inputs=np.dot(self.wih,inputs)
        
        hidden_outputs=self.activation_function(hidden_inputs)
        
        #final input
        final_inputs=np.dot(self.who,hidden_outputs)
        
        #sigmoid function
        final_outputs=self.activation_function(final_inputs)
        
        return final_outputs
        pass
    
pass
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
# learning rate is 0.3
learning_rate = 0.3

help(np.random.rand) #returns array of dim(as a param) with values b/w 0 and 1 
#But this return only positive values so we can subtract 0.5 from it.

#np.random.rand(3,3)-0.5 


# create instance of neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,
learning_rate)    
    
a=n.query([1.0,0.5,-1.5])    
print(a)    

