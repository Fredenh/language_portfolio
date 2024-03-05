import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
from joblib import dump, load
import zipfile

# Creating function that unzips the data to the "in" folder
def unzip_file(zip_file_path, target_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

# Making the correct pathing so that it unzips in the correct location
zip_file_path = os.path.join(".", "in", "archive.zip")
target_folder = os.path.join(".", "in")

# Changing the current working directory to the parent directory of the "in" folder (had to have this in order for it to work)
os.chdir(os.path.join(".", "in", ".."))

# Calling the function to unzip the data
unzip_file(zip_file_path, target_folder)

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
    # Setting up the count vectorizer
    vectorizer = CountVectorizer(ngram_range = (1,2),      
                                lowercase =  True,       
                                max_df = 0.95,           
                                min_df = 0.05,            
                                max_features = 100) 
    # First the training data is fitted to the vectorizer
    X_train_feats = vectorizer.fit_transform(training_data) 
    #... then the test data
    X_test_feats = vectorizer.transform(test_data)
    # Saving the vectorizer to the "models" folder using joblib
    dump(vectorizer, "./models/count_vectorizer.joblib") 
    return X_train_feats, X_test_feats

# Creating function that sets up the lr classifier and initializes it on the training and test features
def lr_model(training_features, test_features, training_labels):
    # Initializing the classifier
    classifier = LogisticRegression(random_state=42).fit(training_features, training_labels)
    # Assigning predictions
    y_pred = classifier.predict(test_features)
    # Saving the classifier to the "models" folder using joblib
    dump(classifier, "./models/LR_classifier.joblib")
    return y_pred

def main():
    # Making the correct pathing so that it unzips in the correct location
    zip_file_path = os.path.join(".", "in", "archive.zip")
    target_folder = os.path.join(".", "in")
    # Changing the current working directory to the parent directory of the "in" folder (had to have this in order for it to work)
    os.chdir(os.path.join(".", "in", ".."))
    # Calling the function to unzip the data
    unzip_file(zip_file_path, target_folder)
    # Load the data
    X_train, X_test, y_train, y_test = load_data()
    # Vectorize the data
    X_train_feats, X_test_feats = doc_vectorizer(X_train, X_test)
    # Run the lr classifier
    y_pred = lr_model(X_train_feats, X_test_feats, y_train)
    # Printing the classifier metrics to the "out" folder
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    folder_path = os.path.join(".", "out")
    # Assigning the name of the report 
    file_name = "logistic_reg_metrics.txt" 
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w") as f: # “writing” classifier report and saving it
        f.write(classifier_metrics)
    
if __name__=="__main__":
    main()