import spacy
import pandas as pd
import re 
import os
import zipfile

# Making a function that unzips the data into the folder "in"
def unzip_file(zip_file_path, target_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

# Creating function that loads and preprocesses the data
def load_text(file_path):
    # Open the text from the file_path as readable 'r'
    with open(file_path, 'r', encoding="latin-1") as file:
        # Assigning the opened text files to the object "text"
        text = file.read()
    # Regex for filtering out the heading and ending of every .txt
    text = re.sub(r'<.*?>', '', text)
    return text

# Creating a function that finds POS and calculate relative frequencies 
def relfreq_doc(text):
    # Loading in the English Language Model from spaCy and applying it to the text
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    # assigning the count of word classes to a zero before counting them with the following for loop
    noun_count = 0
    adjective_count = 0
    verb_count = 0
    adverb_count = 0

    for token in doc: # for every token in doc
        if token.pos_ == "NOUN": # If token is a noun 
            noun_count += 1 # Add 1 to the count
        elif token.pos_ == "ADJ": # The next lines of code carries on with the same principle
            adjective_count += 1
        elif token.pos_ == "VERB":
            verb_count += 1
        elif token.pos_ == "ADV":
            adverb_count += 1

    # Calculating the relative frequency of each word class and rounding to the amount of desired decimals
    relative_freqN = (noun_count / len(doc)) * 10000
    relative_freqN = round(relative_freqN, 2)
    relative_freqAdj = (adjective_count / len(doc)) * 10000
    relative_freqAdj = round(relative_freqAdj, 2)
    relative_freqV = (verb_count / len(doc)) * 10000
    relative_freqV = round(relative_freqV, 2)
    relative_freqAdv = (adverb_count / len(doc)) * 10000
    relative_freqAdv = round(relative_freqAdv, 2)
    
    return doc, relative_freqN, relative_freqAdj, relative_freqV, relative_freqAdv

# Creating function that finds unique entities 
def extract_entities(doc):
    # Making empty lists for the three entities i want to append to the text
    unique_entities_PER = set()
    unique_entities_LOC = set()
    unique_entities_ORG = set()

    for ent in doc.ents: # For entitities in doc.ents
        if ent.label_ == "PER": # If the label = PER, then append it 
            unique_entities_PER.add(ent.text)
        elif ent.label_ == "LOC":
            unique_entities_LOC.add(ent.text)
        elif ent.label_ == "ORG":
            unique_entities_ORG.add(ent.text)

    # Making it into the count of unique entities 
    num_unique_PER = len(unique_entities_PER)
    num_unique_LOC = len(unique_entities_LOC)
    num_unique_ORG = len(unique_entities_ORG)

    return num_unique_PER, num_unique_LOC, num_unique_ORG
    
def process_files(directory):
    # Creating empty object
    all_data = []
    for folder_name in os.listdir(directory): # For folders in path to "USEcorpus"
        folder_path = os.path.join(directory, folder_name) # Assigning path to folders within the "USEcorpus" folder
        if os.path.isdir(folder_path): # Checking whether the path is a directory or not
            # Creating empty object
            data = [] 
            for file_name in os.listdir(folder_path): # For the files in the folders within "USEcorpus"
                file_path = os.path.join(folder_path, file_name) # Assigning path to files within the folders
                if os.path.isfile(file_path): # If what is inside is a file, then do the following lines of code for each of the files
                    text = load_text(file_path) # Using function above to load preprocessed text
                    doc, relative_freqN, relative_freqAdj, relative_freqV, relative_freqAdv = relfreq_doc(text) # Processing the document and calculates relative frequencies of different POS tags
                    num_unique_PER, num_unique_LOC, num_unique_ORG = extract_entities(doc) # Extracting the unique entities from all the files
                    # Creating a pandas dataframe with the appended data and assigning the label names 
                    final_data = pd.DataFrame([(file_name, relative_freqN, relative_freqV, relative_freqAdj,
                                                relative_freqAdv, num_unique_PER, num_unique_LOC, num_unique_ORG)],
                                              columns=["filename", "RelFreq NOUN", "RelFreq ADJ", "RelFreq VERB",
                                                       "RelFreq ADV", "Unique PER", "Unique LOC", "Unique ORG"])
                    data.append(final_data) # Appending the final data to the empty object i created above
            all_data.extend(data) # Extending the empty object i created in the for loop above
            save_data(data, folder_name) # Calling function below

# Creating a function that saves the pd dataframe to a csv with a fodler that corresponds to the given folder
def save_data(data, folder_name):
    ult_data = pd.concat(data)
    outpath = os.path.join(".", "out", folder_name + ".csv")
    ult_data.to_csv(outpath, index=False)

def main():
    # Making the correct pathing so that it unzips in the correct location
    zip_file_path = os.path.join(".", "in", "USEcorpus.zip")
    target_folder = os.path.join(".", "in")
    # Changing the current working directory to the parent directory of the "in" folder (had to have this in order for it to work)
    os.chdir(os.path.join(".", "in", ".."))
    # Calling the function to unzip the data
    unzip_file(zip_file_path, target_folder)
    # Providing the path to the USEcorpus from which the file processing starts
    directory = os.path.join(".", "in", "USEcorpus")
    process_files(directory)

if __name__ == "__main__":
    main()
