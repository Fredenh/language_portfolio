import string
import os 
import pandas as pd
import numpy as np
import argparse
import random
import tensorflow as tf
# Setting random seeds
tf.random.set_seed(42)
np.random.seed(42)
import warnings
import sys
sys.path.append(".")
import utils.requirements_functions as rf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Function that loads and preprocesses data
def load_data(data_dir, sample_size):
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
    # Utilizing the random seed from before on the data
    np.random.shuffle(all_comments)
    # Randomizing the subset data
    subset_comments = all_comments[:sample_size]
    # Cleaning the strings and taking out intergers in the process
    corpus = [rf.clean_text(x) if isinstance(x, str) else "" for x in subset_comments]
    return corpus

# Creating function that tokenizes the data and generates padded sequences
def tokenize_data(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus)
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences, total_words)
    return tokenizer, predictors, label, max_sequence_len, total_words

# Creating function that generates text with rf.generate_text from utils
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    generated_text = rf.generate_text(seed_text, next_words, model, tokenizer, max_sequence_len)
    return generated_text

def main():
    # Argparse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join(".", "models", "trained_model.tf"))
    parser.add_argument('--seed_text', type=str, default='Enter your seed text here')
    parser.add_argument('--next_words', type=int, default=10)
    args = parser.parse_args()
    # Loading the saved model
    model = tf.keras.models.load_model(args.model_path)
    # Loading and preprocessing the data
    data_dir = os.path.join(".", "data", "news_data")
    sample_size = 1000
    corpus = load_data(data_dir, sample_size)
    tokenizer, predictors, label, max_sequence_len, total_words = tokenize_data(corpus)
    # Generating text based on the args 
    generated_text = generate_text(args.seed_text, args.next_words, model, tokenizer, max_sequence_len)
    print(generated_text)

if __name__ == '__main__':
    main()
