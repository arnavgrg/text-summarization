from __future__ import print_function
import pandas as pd
import keras
from keras.models import Model, save_model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np
import re
from pickle import dump, load
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import seaborn as sns
import math
import tensorflow as tf

#Enable program to use GPU
# config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

#Load processed dataset from data_cleaning.py
data = load(open('processed_dataset.pkl', 'rb'))
print('Loaded {} samples from processed_dataset.pkl'.format(len(data)))

#Hyperparameters from training
batch_size = 64
epochs = 50
latent_dim = 256
num_samples = 25000

#Process text to create training data needed for the model
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
for obj in data[:min(num_samples, len(data)-1)]:
    input_text = obj['text']
    #\t serves as starting sequence character and \n is ending sequence character
    target_text = '\t' + obj['summary'] + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    #This will server as the number of inputs for the model
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    #This will serve as the number of outputs for the model
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

#Calculate additional data and lists needed for training
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
encoder_seq_length = [len(txt) for txt in input_texts]
decoder_seq_length = [len(txt) for txt in target_texts]
max_encoder_seq_length = max(encoder_seq_length)
max_decoder_seq_length = max(decoder_seq_length)

#Print out statistics
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

#Generic function to create training encoder and decoder, 
#as well as inference encoder and decoder 
def create_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    # compiled model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,  initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

#Call create_models function using n_inputs = num_encoder_tokens, n_outputs = num_decoder_tokens and 256 LSTM cells
model, encoder_model, decoder_model = create_models(num_encoder_tokens, num_decoder_tokens, latent_dim)

#Create empty 3 dimensional matrices, which is essentially a two-dimensional
#vector for each input sample, where the columns are all the possible input tokens, 
#and the rows are the actual letters that appear in the input sentence (one hot encoded)
#against the possible input tokens.
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

#Create dictionaries that map each unique character to a number
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

#Actually create the one hot encoded vector for each item in the input and target datasets
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

#Compile the model - set the configuration parameters
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#Train the model
print("Starting training...")
model.fit([encoder_input_data, decoder_input_data], 
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
print("Finished training!")

#Save the model's weights
print("Saving model weights...")
save_model(model, 'summarizer-weights.h5')
print("Model weights saved")

#Trying saving encoders weights
try:
    print("Saving encoder model...")
    save_model(encoder_model, 'encoder-weights.h5')
    print("Encoder weights saved")
except Exception as e:
    f = open('error_log.txt', 'w+')
    f.write("Error while saving encoder model: {}".format(str(e)))
    f.write("\n")
    f.close()

#Try saving decoders weights
try:
    print("Saving encoder model...")
    save_model(decoder_model, 'decoder-weights.h5')
    print("Encoder weights saved")
except Exception as e:
    f = open('error_log.txt', 'w+')
    f.write("Error while saving decoder model: {}".format(str(e)))
    f.write("\n")
    f.close()

#Load the saved model
# print("Loading saved model...")
# model = load_model('summarizer-weights.h5')
# print("Loaded saved summarizer model!!")

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# Function to decode the sequence
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [h, c]
    return decoded_sentence

#See output predictions for unseen data
f = open("./test-cases/t2.txt", "w+")
for index in range(500):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[index:index+1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[index])
    print('Decoded sentence:', decoded_sentence)
    f.write("Input sentence: {}\n".format(input_texts[index]))
    f.write("Decoded sentence: {}\n".format(decoded_sentence))
f.close()