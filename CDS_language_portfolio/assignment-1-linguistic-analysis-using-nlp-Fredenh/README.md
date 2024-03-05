[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10145302&assignment_repo_type=AssignmentRepo)
# Assignment 1 - Extracting linguistic features using spaCy
This is the first of five assignment for the Language Analytics course.

# Contribution
This assignment was initially done in collaboration with other students from the course. I also consulted the notebooks from class for help. The final version of the code is done by myself. This includes the process of converting from a notebook to a script and troubleshooting some minor errors from the initial notebook version.

# Ross' instructions
This assignment concerns using ```spaCy``` to extract linguistic information from a corpus of texts.

The corpus is an interesting one: *The Uppsala Student English Corpus (USE)*. All of the data is included in the folder called ```in``` but you can access more documentation via [this link](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457).

For this exercise, you should write some code which does the following:

- Loop over each text file in the folder called ```in```
- Extract the following information:
    - Relative frequency of Nouns, Verbs, Adjective, and Adverbs per 10,000 words
    - Total number of *unique* PER, LOC, ORGS
- For each sub-folder (a1, a2, a3, ...) save a table which shows the following information:

|Filename|RelFreq NOUN|RelFreq VERB|RelFreq ADJ|RelFreq ADV|Unique PER|Unique LOC|Unique ORG|
|---|---|---|---|---|---|---|---|
|file1.txt|---|---|---|---|---|---|---|
|file2.txt|---|---|---|---|---|---|---|
|etc|---|---|---|---|---|---|---|

## Objective

This assignment is designed to test that you can:

1. Work with multiple input data arranged hierarchically in folders;
2. Use ```spaCy``` to extract linguistic information from text data;
3. Save those results in a clear way which can be shared or used for future analysis

## Some notes
- The data is arranged in various subfolders related to their content (see the [README](in/README.md) for more info). You'll need to think a little bit about how to do this. You should be able do it using a combination of things we've already looked at, such as ```os.listdir()```, ```os.path.join()```, and for loops.
- The text files contain some extra information that such as document ID and other metadata that occurs between pointed brackets ```<>```. Make sure to remove these as part of your preprocessing steps!
- There are 14 subfolders (a1, a2, a3, etc), so when completed the folder ```out``` should have 14 CSV files.

# Data
The data for this assignment is [_The Uppsala Student English Corpus (USE)_](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457) which was gathered by [Ylva Berglund and Margareta Westergren Axelsson](https://www.engelska.uu.se/research/english-language/electronic-resources/use/). It is a collection of 1489 essays written by 440 Swedish university students of English at three different levels of terms. The data was gathered between 1999 and 2001. Additional information on the data can be found within the readme file inside the _in_ folder.In this assignment, the data is downloaded as a zip file and unzipped from within the python pipeline as you run it. 

# Packages
I used a variety of packages in order to solve this assignment. The following is a list of the packages i used and for what purpose they were utilized.
* ```spaCy``` i used this package to extract linguistic features from USE corpus.
* ```pandas``` i used pandas to create a dataframe before saving the output to a CSV.
* ```re``` i used this package to create a regular expression that would filter out any additional information i did not want.
* ```os``` i used this package to navigate paths.
* ```zipfile``` i used this package to create a function that unzips the data in the desired location.

# Methods
This assignment has one python script that tackles the task at hand. It is called code.py and is located in the _src_ folder. The script starts by unzipping the data in the _in_ folder by utilizing the ```zipfile``` package. Then the unzipped data is loaded in and a regular expression is created using ```re.sub()``` which filters out undesired characters from the .txt files. Then the "en_core_web_md" English language model provided by ```spaCy``` is loaded in and applied to the USE corpus. Then the script runs a for loop that counts the amount of nouns, adjectives, verbs and adverbs before calculating the relative frequency of them all per 10000 words in the corpus. Then the script proceeds to extract unique entities in the corpus. It does this by running a for loop that identifies the "PER", "LOC" and "ORG" entities and counts them afterwards. Then the script runs a new for loop that iterates through the folders and files in order to create a ```pandas``` dataframe with the relative frequencies of the parts of speech as well as the unique entities of every .txt file in every subfolder. The result is a CSV for every subfolder which is then saved to the _out_ folder.

# Discussion of results
The results of this pipeline gives a shallow insight into the grammatical structure of foreign English speakers studying English at university across three different terms. Before consulting the output, one might suggest that the amount of unique entities increases as the students progress in their degrees. Just from briefly looking at the CSV's, it is clear that there definetely is a difference in the amount of entities between the first term students and the second and third term students. However between the second and third term students, the discrepancy is hard to spot.

# Usage 
* First you need to acquire the data from the [Oxford Text Archive](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457) and place the zip file in the _in_ folder.
* Then download the _en_core_web_md_ by running this from the command line: ```python3 -m spacy download en_core_web_md```
* The you run ```bash setup.sh``` from the command line. This installs requirements and creates a virtual environment.
* Run ```source ./assignment1_env/bin/activate``` 
* Then run the script ```python3 src/code.py```. Included in this sript is the unzipping of the data into the _in_ folder.
* After running the script, the CSV's are located in the _out_ folder.
