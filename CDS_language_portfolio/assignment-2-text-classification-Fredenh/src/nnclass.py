import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
from joblib import dump, load

# Creating a function that loads splits it into train and test splits
def load_data():
    # Loading data
    filename = os.path.join(".", "in", "fake_or_real_news.csv")
    data = pd.read_csv(filename, index_col=0)
    # Getting text and labels from the columns in the dataframe
    X = data["text"]
    y = data["label"]
    # Makig train and test split with a 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,          
                                                        test_size=0.2,   
                                                        random_state=42) 
    return X_train, X_test, y_train, y_test

# Creating a function that vectorizes the train and test data
def doc_vectorizer(training_data, test_data):    
    # Setting up the vectorizer with the Tfidf
    vectorizer = TfidfVectorizer(ngram_range = (1,2), # Unigrams and bigrams (1 word and 2 word units) // including single tokens and also two word tokens like "New York"
                             lowercase =  True,       
                             max_df = 0.95,           # Removing very common words and removing stop words. 
                             min_df = 0.05,           # Removing very rare words / extreme outliers // words that occur too scarcely so they dont have an impact
                             max_features = 500) 
    # First the training data is fitted to the vectorizer
    X_train_feats = vectorizer.fit_transform(training_data) 
    #... then the test data
    X_test_feats = vectorizer.transform(test_data) 
    # Saving the vectorizer to the "models" folder using joblib
    dump(vectorizer, "./models/tfidf_vectorizer.joblib") 
    return X_train_feats, X_test_feats

# Creating function that sets up the nn classifier and initializes it on the training and test features
def nn_model(training_features, test_features, training_labels):
    # Initializing the classifier
    classifier = MLPClassifier(activation="logistic", # = sigmoid that maps the input values between 0 and 1
                               hidden_layer_sizes=(20,), # Number of neurons in each hidden layer
                               max_iter=1000, # Setting a max of 1000 iterations (epochs)
                               random_state=42) 
    # Fitting the model
    classifier.fit(training_features, training_labels)
    # Assigning predictions
    y_pred = classifier.predict(test_features)
    # Saving the classifier to the "models" folder using joblib
    dump(classifier, "./models/NN_classifier.joblib")
    return y_pred

def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_data()
    # Vectorize the data
    X_train_feats, X_test_feats = doc_vectorizer(X_train, X_test)
    # Run the nn classifier
    y_pred = nn_model(X_train_feats, X_test_feats, y_train)  # Updated argument names
    # Printing the classifier metrics to the "out" folder
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    folder_path = os.path.join(".", "out")
    # Assigning the name of the report 
    file_name = "neural_net_metrics.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w") as f: # “writing” classifier report and saving it
        f.write(classifier_metrics)

if __name__=="__main__":
    main()