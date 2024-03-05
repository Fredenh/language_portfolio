import string
import os 
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import warnings
import sys
import zipfile
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# Setting random seeds
np.random.seed(42)
tf.random.set_seed(42)
sys.path.append(".")
import utils.requirements_functions as rf

# Creating function that unzips the data to the "news_data" folder
def unzip_file(zip_file_path, target_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

# Creating a function that loads the data 
def load_data(data_dir):
    # Empty object
    all_comments = []
    # For files in the "news_data" folder
    for filename in os.listdir(data_dir):
        # If there is "Comments" in the filename...
        if 'Comments' in filename:
            # ... Then read to a dataframe 
            comment_df = pd.read_csv(os.path.join(data_dir, filename))
            # Then identifying which column in the dataframe that is needed and extending it to the empty object from before
            all_comments.extend(list(comment_df["commentBody"].values))
    return all_comments

# Creating function that randomizes the data and cleans it
def randomize_data(data):
    # Specifying the desired size of the random sample
    sample_size = 1000  
    # Utilizing the random seed from before on the data
    np.random.shuffle(data)
    # Randomizing the subset data
    subset_data = data[:sample_size]
    # Cleaning the strings and taking out intergers in the process
    corpus = [rf.clean_text(x) if isinstance(x, str) else "" for x in subset_data]
    return subset_data, corpus

# Creating function that tokenizes the data and generates padded sequences
def tokenize_data(corpus):
    # Creating a vocabulary while tokenizing it
    tokenizer = Tokenizer() 
    # Fitting the tokenizer on the randomized and cleaned text
    tokenizer.fit_on_texts(corpus)
    # Calculating total number of words that have been tokenized
    total_words = len(tokenizer.word_index) + 1 
    # Last preprocessing before training using functions from the "requirements_functions.py" script
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus)
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences, total_words)
    
    return tokenizer, predictors, label, max_sequence_len, total_words

# Creating function that trains the model on the randomized and tokenized data
def train_model(max_sequence_len, total_words, predictors, label):
    # Using helper function from utils to create the model
    model = rf.create_model(max_sequence_len, total_words)
    # Defining which parameters the model has
    history = model.fit(predictors, 
                        label, 
                        epochs=100, # Number of epochs
                        batch_size=128, # Large batch size for the purpose of speeding up training
                        verbose=1)
    # Assigning the correct path and name for the model.save function
    folder_path = os.path.join(".", "models")
    file_name = "trained_model.tf"
    file_path = os.path.join(folder_path, file_name)
    #model.save(file_path, overwrite=True, save_format="tf")
    return model

def main():
    # Data processing tools
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Unzipping the data to correct folder
    zip_file_path = os.path.join(".", "data", "news_data", "archive.zip")
    target_folder = os.path.join(".", "data", "news_data")
    # Change the current working directory to the parent directory of the "in" folder (has to do this to make it work)
    os.chdir(os.path.join(".", "data", ".."))
    # Calling the function to unzip the file
    unzip_file(zip_file_path, target_folder)
    # Assigning path for the data directory
    data_dir = os.path.join(".", "data", "news_data")
    all_comments = load_data(data_dir)
    subset_comments, corpus = randomize_data(all_comments)
    tokenizer, predictors, label, max_sequence_len, total_words = tokenize_data(corpus)
    train_model(max_sequence_len, total_words, predictors, label)
    model = train_model(max_sequence_len, total_words, predictors, label)
    # Testing that the pipeline worked
    print(rf.generate_text("english", 10, model, tokenizer, max_sequence_len))

if __name__ == '__main__':
    main()
