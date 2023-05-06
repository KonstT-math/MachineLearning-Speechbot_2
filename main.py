
# project: chatbot with speech recognition via DNN

# modules to download: 
# numpy,
# nltk
# for speech to text
# vosk, pyaudio
# for text to speech
# pyttsx3

# 1) chatbot with machine learning
# 2) enrich with speech recognition via VOSK API

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
#import speech_recognition as sr
import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio

# ----------------------------------------
# initialization

# stemming is the process of reducing a word to its word stem. Word stem is a base or root form of the word and doesn’t need to be an existing word. For example, the Porter algorithm reduces the words “argue”, “argued”, “argues” and “arguing” to the stem “argu” which isn’t an existing word.
stemmer = PorterStemmer()

# open and read intents.json:
with open("intents.json") as file:
	data = json.load(file)

# ----------------------------------------
# for predictions

#def mse(y_true, y_pred):
#	return np.mean(np.power(y_true-y_pred, 2));

def ReLU(Z):
	return np.maximum(0,Z)

#def d_ReLU(Z):
#	return Z>0

def tanh(x):
	return np.tanh(x);

#def d_tanh(x):
#	return 1-np.tanh(x)**2;

def sig(x):
	return 1/(1 + np.exp(-x))

#def d_sig(x):
#	return sig(x)*(1- sig(x))

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]
	
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
	
	return np.array(bag)

# describe the model and its architecture (as in the train file)
# model: DNN with 1 hidden layer
def predict(W1, b1, W2, b2, X):
	Z1 = np.dot(X,W1) + b1 
	A1 = ReLU(Z1)
	Z2 = np.dot(A1,W2) +b2
	A2 = Z2
	return Z1, A1, Z2, A2

# ----------------------------------------
# retrieve weights and biases from csv files:

W1 = pd.read_csv('L1weight1.csv')
b1 = pd.read_csv('L1bias1.csv')
W2 = pd.read_csv('L1weight2.csv')
b2 = pd.read_csv('L1bias2.csv')

W1 = W1.to_numpy()
b1 = b1.to_numpy()
W2 = W2.to_numpy()
b2 = b2.to_numpy()

# retrieve words, labels, sd_out, m_out from pickle file
with open("data.pickle", "rb") as f:
	words, labels, m_out, sd_out = pickle.load(f)

# ----------------------------------------
# function used for "speak" mode
# listens speech via VOSK API, and uses the trained model to respond

# for listening:
model = Model('path_to_vosk_model')
recognizer = KaldiRecognizer(model, 16000)
mic = pyaudio.PyAudio()
stream = mic.open(format = pyaudio.paInt16, channels=1, rate = 16000, input=True, frames_per_buffer=8192)

# for speaking:
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-60)

def speak_vosk(listen):
	if listen==1:
		stream.start_stream()
		while True:
			data = stream.read(4096)
			#if len(data)==0:
			#	break
			if recognizer.AcceptWaveform(data):
				text = recognizer.Result()
				a = json.loads(text)
				inp = a.get('text')

				return inp

def chat_speakmode():

	print("Start talking with the bot! (say \"shut down\" to stop)")
	
	while True:
		
		print("You: ")
		inp = speak_vosk(1)
		if inp =='shut down':
			break
		print(inp)
		results = predict(W1, b1, W2, b2, [bag_of_words(inp, words)])
		#print([bag_of_words(inp, words)])
		# denormalize results
		response = results[3]*sd_out + m_out
		# get index corresponding to json file for predicted response
		response_index = round(np.asscalar(response))
		#print(response)
		# put the index to corresponding label in json file
		tag = labels[response_index]
		#print(tag)
		
		# stop listening
		speak_vosk(0)

		for tg in data["intents"]:
			if tg["tag"] == tag:
				responses = tg["responses"]
		a_response = random.choice(responses)
		print("Abott: {}".format(a_response))
		engine.say(a_response)
		engine.runAndWait()

		# start listening
		speak_vosk(1)
	

		


# ----------------------------------------
# main program

chat_speakmode()

# ----------------------------------------

