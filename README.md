# MachineLearning-Speechbot_2

A Machine Learning implemented Speech Bot, which listens via VOSK API (works offline) and responds via espeak and pyttsx3 library. It uses a neural network with one hidden layer, implemented from scratch.

Implemented in Python.

The install.sh is a shell script containing the main Python modules that are needed for the setup. It creates a virtual environment named abott_env. This is being done via virtualenvwrapper. Linux Ubuntu 18.04.6 LTS.

The json file intents.json can be filled with patterns and responses in order to be used for chat

Run the training.py in order to train the model. A DNN with 1 fully connected hidden layers is implemented from scratch. It creates csv files with weights and biases and a pickle file with words, labels and output parameters; these are used in main.py for responses.

The main.py contains the main algorithm.
