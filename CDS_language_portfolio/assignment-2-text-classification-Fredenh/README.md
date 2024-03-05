# Assignment 2 - Text classification benchmarks
This is the second of five assignments for the Language Analytics course.

# Contribution
The assignment was initially done in collaboration with other course participants. I also used the notebooks from the in-class sessions for help with compiling a functioning pipeline. The final version of the code was developed independently by me.  

# Ross' instructions
This assignment is about using ```scikit-learn``` to train simple (binary) classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## Objective

This assignment is designed to test that you can:

1. Train simple benchmark machine learning classifiers on structured text data;
2. Produce understandable outputs and trained models which can be reused;
3. Save those results in a clear way which can be shared or used for future analysis

# Data 
The data used for this assignment is the [Fake or Real News dataset](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) which was gathered on an unspecified open source website by Kaggle user Jillani Soft Tech. When unzipped, the dataset consists of a CSV with 4 columns: index, title, text and label. This assignment works with the labels FAKE or REAL in order to train a logistic regression classifier and Neural network classifier. The dataset will be unzipped from within the _logreg.py_ script.

# Packages
For this assignment i used a variety of different packages for different purposes. I will run through them in the following bullet point list.
* ```os``` is used to navigate paths
* ```pandas``` is used to read the data into a ```pandas``` object from a CSV
* ```scikit-learn``` is used to train the two classification models as well as vectorizing 
* ```joblib``` is used to save the vectorizers and classifiers for later reuse
* ```zipfile``` is used to unzip the data 

# Methods
For this assignment there a two scripts, _logreg.py_ and _nnclass.py_, that follow the same principles in terms of pipeline. However, they differantiate in the vectorizers they use. The _logreg.py_ script uses a ```CountVectorizer()``` whereas the _nnclass.py_ uses a ```TfidfVectorizer()``` from ```scikit-learn```. Also in terms of classification models, there is a difference in the two scripts. _logreg.py_ uses a logistic regression classifier whereas the _nnclass.py_ script uses a neural network classifier. 
Both the scripts loads the data and creates train and test splits using ```train_test_split()``` from ```scikit-learn```. Then the two different vectorizers are initialized and fitted to the train and test data. Then they are saved to the folder called _models_ by using the ```dump()``` method from ```joblib```. Then the two different classifiers are initialized and trained on the data before they are saved to the _models_ folder using the same principle as described above. Then the script creates and prints two classification metrics, one for each classification model, which are stored in the _out_ folder. 

# Discussion of results
While consulting the output of the trained classifiers in the _out_ folder, it is clear that the neural network classifier performed better than the logistic regression classifier. The neural network classifier had an f1-score accuracy score of 0.89 which is 0.7 higher than the 0.82 of the logistic regression classifier. There are different reasons to this. One of them is that the neural network classifier has the advantage of having the ability to capture non-linear relationships in the data whereas, the logistic regression classifier, on the other hand, has a more linear approach to classifying the data as it assumes that there is a linear relationship between the features in the data. But overall, it can be said that both models performed well on the data. Both scores are acceptable and suggest that they was largely succesfull in classifying Real or Fake news.

# Usage
* First you need to acquire the data from [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) and place the zip file in the _in_ folder.
* Then you run ```bash setup.sh``` from the command line to install packages and create a virtual environment
* Run ```source ./assignment2_env/bin/activate``` to activate the virtual environment
* !OBS! it is important you run the _logreg.py_ script first ```python3 src/logreg.py``` because it is the script that unzips the data
* Then run the second script ```python3 src/nnclass.py``` 
* The output of the scripts are located in the _out_ folder

