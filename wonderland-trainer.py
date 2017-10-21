# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:30:39 2017

@author: KOGILVIE
Based on tutorial for LSTM RNN at
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras


Alice in Wonderland LSTM RNN for text generation
This script trains the LSTM RNN. It is meant
to be used with the wonderland-generator.py script,
which actually generates the text once the model
has been trained. 
"""

import numpy
import pickle
import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and convert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
text_tokens = nltk.wordpunct_tokenize(raw_text)
# create mapping of unique chars to integers
chars = sorted(list(set(text_tokens)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# now we summarize the data set
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters", n_chars)
print("Total Vocab (unique chars)", n_vocab)


'''
    We will split the book text up into subsequences of length 100. 
    The length in this case is arbitraty. Each training pattern of the network
    is comprised of 100 time steps of one character input (X) followed by one 
    character output (y). When creating these sequences, we slide this 
    window along the whole book one character at a time, allowing each
    character a chance to be learned from the 100 characters that preceeded it 
    (except for the first 100 characters, of course).
'''

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns", n_patterns)


'''
    Now we need to transform the data so that it can be used with Keras. 
    1. First we must transform the list of input sequences into the form
    [samples, time steps, features] expected by an LSTM network. 
    2. Next we need to rescale the integers to the rance 0 to 1 to make
    the patterns easier to learn by the LSTM network that uses the sigmoid 
    activation function by default. 
    3. Finally, we need to convert the output patterns 
    (single characters converted to integers) into a one hot encoding. 
    This is so that we can configure the network to predict the probability
    of each of the 47 different characters in the vocabulary rather than
    trying to force it to predict precisely the next character. Each y 
    value is converted into a sparse vector with length 47, full of zeros
    except with a 1 in the column for the letter (integer) that the pattern 
    represents. 
    
    For example, "n", which is integer 31, would be one hot encoded as:
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.]   
'''

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


''' 
    Now we can define our LSTM model. Here we define a single hidden 
    LSTM layer with 256 memory units. The network uses droupout with a 
    probability of 20. The output layer is a Dense layer using the softmax
    activation function to output a probability prediction for each of the 47
    characters between 0 and 1.
    This is basically a single character classification problem with 47 classes
    and as such is defined as optimizing the log loss (cross entropy) using
    the ADAM optimization algorithm for speed.
'''

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0,2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


''' 
    We're not interested in accuracy, because if we were really accurate 
    we'd just predict the characters exactly as they appear. We're interested
    instead in generalization of the dataset that minimizes the chosen loss
    function. We are seeking a balance between generalization and overfitting
    but short of memorization. 
'''

'''
    The model is very slow to train, so we will use model checkpointing to 
    record all of the network weights to file each time an improvement in loss 
    is observed at the end of the epoch. We will use the best set of weights 
    (lowest loss) to instantiate our generative model in the next section. 
'''

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint= ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list=[checkpoint]

# fit the model to the data, using 20 epochs and 128 patterns
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

pickle.dump(model, open("model.p", "wb"))
pickle.dump(chars, open("chars.p", "wb"))
pickle.dump(dataX, open("dataX.p", "wb"))
pickle.dump(n_vocab, open("n_vocab.p", "wb"))
