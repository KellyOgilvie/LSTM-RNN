# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:31:19 2017

@author: KOGILVIE
Based on tutorial for LSTM RNN at
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras


Alice in Wonderland LSTM RNN for text generation
This script generates text based on the model trained
in the wonderland.py script. 

"""

import numpy
import pickle
import sys


# recover the model from file
model = pickle.load(open("model.p", "rb"))
chars = pickle.load(open("chars.p", "rb"))
dataX = pickle.load(open("dataX.p", "rb"))
n_vocab = pickle.load(open("n_vocab.p", "rb"))

# load the network weights
filename = "weights-improvement-xx-x-xxxx.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# create a reverse mapping for converting integers back to characters
int_to_char = dict((i, c) for i, c in enumerate(chars))

''' 
    Finally, we can begin making actual predictions. 
    THe simplest way to use Keras LSTM model to make predictions is to 
    start off with a seed sequence as input, generate the next character, 
    then update the seed sequence to add the generated character on the end 
    and trim off the first character. This process is repeated for as long
    as we want to predict new characters.
'''

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:", "\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")
