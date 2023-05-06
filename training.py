
# project: chatbot with speech recognition via DNN

# modules to download: 
# numpy
# pandas
# nltk

# 1) chatbot with machine learning
# 2) enrich with speech recognition via google api

# ----------------------------------------
# imports

import numpy as np
import pandas as pd
import nltk
# also, get into a python shell and type:
# >>> import nltk
# >>> nltk.download('punkt')
from nltk.stem import PorterStemmer
# from nltk.stem.lancaster import LancasterStemmer
import random
import json
import pickle

# ----------------------------------------
# initialization

# stemming is the process of reducing a word to its word stem. Word stem is a base or root form of the word and doesn’t need to be an existing word. For example, the Porter algorithm reduces the words “argue”, “argued”, “argues” and “arguing” to the stem “argu” which isn’t an existing word.
stemmer = PorterStemmer()

# opens the intents.json file with the patterns and responces
# and saves it to data
with open("intents.json") as file:
	data = json.load(file)

# will keep the tokenized words in words
words = []
# will keep "tag"s in labels
labels = []
docs_x = []
docs_y = []

# runs through "intents"
for intent in data["intents"]:
	# runs through "patterns" in "intents"
	for pattern in intent["patterns"]:
		# tokenizes the words in "patterns"
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		# stores the tokenized word in docs_x
		docs_x.append(wrds)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

# normalize and sort words in the following way:
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

# sort labels
labels = sorted(labels)

# ----------------------------------------
# preprocessing

# up to now, we have strings
# neural networks work with numbers.. this is our next step..

# encode words: for input/ouput to neural network
# [0,1,0,0,0,1,1,0,0,0,1,1,0] - if exists (1) or not (0)
# example for "greeting":
# [1,1,0,0]
# "hi", "hey", "sell", "help"

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
	bag = []
	# stem words in doc_x
	wrds = [stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(labels.index(docs_y[x]))

#print(output)
# save output in file for use in main  

# ----------------------------------------
# training and testing output

training = np.array(training)

output = np.array(output).reshape(1,-1)
output = output.T
# normalize output to be with mean 0 and std 1
m_out = np.mean(output)
sd_out = np.std(output)
output = (output - m_out)/sd_out

# ----------------------------------------
# the training model

nofeats = len(training[0])
nod1 = 10 # number of nodes for hidden layer 1

# activation functions:
def mse(y_true, y_pred):
	return np.mean(np.power(y_true-y_pred, 2));

def ReLU(Z):
	return np.maximum(0,Z)

def d_ReLU(Z):
	return Z>0

def tanh(x):
    return np.tanh(x);

def d_tanh(x):
    return 1-np.tanh(x)**2;

def sig(x):
    return 1/(1 + np.exp(-x))

def d_sig(x):
    return sig(x)*(1- sig(x))

def init_weights():
	W1 = np.random.rand(nofeats,nod1)-0.5 
	b1 = np.random.rand(1,nod1)-0.5 

	W2 = np.random.rand(nod1,1)-0.5 
	b2 = np.random.rand(1,1)-0.5 
	return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
	Z1 = np.dot(X,W1) + b1 
	A1 = ReLU(Z1)
	Z2 = np.dot(A1,W2) +b2
	A2 = Z2
	return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
	m = Y.size
	dZ2 = (A2 - Y) 
	dW2 = (1/m)*np.dot(A1.T, dZ2)
	db2 = np.mean(dZ2,axis=0, keepdims=True) 

	dZ1 = np.dot(dZ2, W2.T) * d_ReLU(A1) 
	dW1 = (1/m)*np.dot(X.T, dZ1) 
	db1 = np.mean(dZ1, axis=0, keepdims=True)

	return dW1, db1, dW2, db2

def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
	W1 = W1 - alpha*dW1 
	b1 = b1 - alpha*db1 
	W2 = W2 - alpha*dW2 
	b2 = b2 - alpha*db2 
	return W1, b1, W2, b2

def nn(X, Y, epochs, alpha):
	W1, b1, W2, b2 = init_weights()	
	for i in range(epochs):
		# forward propagation
		Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
		loss = mse(A2, Y)
		if i%100 == 0:
			print("loss at iteration" , i, loss)		
		# backward propagation
		dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
		W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
		
	return W1, b1, W2, b2

W1, b1, W2, b2 = nn(training, output, 50000, 0.01)

# ----------------------------------------
# dump output variables to files:

# create csv files with current weights and biases
dfW1 = pd.DataFrame(W1)
dfW1.to_csv(r'L1weight1.csv', index = False, header=True)

dfb1 = pd.DataFrame(b1)
dfb1.to_csv(r'L1bias1.csv', index = False, header=True)

dfW2 = pd.DataFrame(W2)
dfW2.to_csv(r'L1weight2.csv', index = False, header=True)

dfb2 = pd.DataFrame(b2)
dfb2.to_csv(r'L1bias2.csv', index = False, header=True)

# write bytes, as seen below, in data.pickle file
with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, m_out, sd_out),f)


# ----------------------------------------

