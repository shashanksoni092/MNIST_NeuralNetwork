#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:42:05 2019
@author: shashanksoni092
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
#Declaring a class named NeuralNetwork( a Structure)
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
        
        self.who +=self.lr*np.dot((output_errors*final_outputs*(1-final_outputs))
        ,np.transpose(hidden_outputs))
        
        self.wih +=self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs))
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
    
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
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


#len(training_mnist_list)
#training_mnist_list[0]

#all_values=training_mnist_list.split(',')
#all_values

#numpy.asfarray() converts text to numbers

#image_array=np.asfarray(all_values[1:]).reshape((28,28))

#plt.imshow(image_array,cmap='Greys',interpolation='None')
    
#(Rescaling the color value (0-255) to (0.01-1) bcoz big values will kill weight updates) or normalizing

#inputs=(np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01 
#targets=np.zeros(output_nodes) +0.01

#targets[int(all_values[0])]=0.99

#n.train(inputs,targets)


pass

#load test data


#all_values=test_mnist_list[0].split(',')
#print(all_values[0])

#image_array=np.asfarray(all_values[1:]).reshape((28,28))
#plt.imshow(image_array,cmap='Greys',interpolation='None')

#n.query((np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01)

#importing training files
training_mnist_file=open("/home/shashanksoni092/DataScienceProjects/Scratch/NeuralNetwork/MNIST/mnist_train_100.csv",'r')
training_mnist_list=training_mnist_file.readlines()
training_mnist_file.close()


#training
epochs = 15
for e in range(epochs):
# go through all records in the training data set
    for record in training_mnist_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        n.train(inputs, targets)
        


#testing
test_mnist_file=open("/home/shashanksoni092/DataScienceProjects/Scratch/NeuralNetwork/MNIST/mnist_test_10.csv",'r')
test_mnist_list=test_mnist_file.readlines()
test_mnist_file.close()
        
# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set
for record in test_mnist_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)