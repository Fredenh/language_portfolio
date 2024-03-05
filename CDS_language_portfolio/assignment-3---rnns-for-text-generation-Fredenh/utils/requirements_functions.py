# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower() # stripping all the punctuation and making it lowercase
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences, num_classes):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest one
    input_sequences = np.array(pad_sequences(input_sequences, 
                                             maxlen=max_sequence_len, 
                                             padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=num_classes)
    return predictors, label, max_sequence_len


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1)) # dropout(0.1) = during training, everytime u do an iteration, 10 percent of those weights should be removed. reason: by constraining it makes the model learner better by making it more difficult for the model. then overfitting is prevented
    
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    for _ in range(next_words): # x in
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list), 
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()