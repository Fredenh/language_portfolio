import pandas as pd
import matplotlib.pyplot as plt
import os
from transformers import pipeline
import zipfile

# Creating function that unzips the data to the "in" folder
def unzip_file(zip_file_path, target_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

# Creating function that loads the data and filters it
def load_and_filter(folder_path, label):
    # Loading data as pd object
    data = pd.read_csv(folder_path)
    # Filtering out only the label column
    filtered_data = data[data['label'] == label]
    return filtered_data

# Creating function that initializes the pipeline from Huggingface
def classify_emotions(texts):
    classifier = pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base",
                          return_all_scores=False)
    # List comprehension that applies the classifier to each line under the label column
    emotions = [classifier(line)[0]["label"] for line in texts]
    return emotions

# Creating function loops through the data and appends emotion count 
def emotion_counter(emotion_list):
    # Empty dictionary
    emotion_count = {}
    # For emotion in emotion list
    for emotion in emotion_list:
        # Checking if the emotion is already in the dictionary, If it is, increment its count by 1. # If it is not, initialize its count to 0 and then increment by 1
        emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
        # Sorting the emotion_count dictionary by key and converting it to a sorted dictionary
    sorted_emotion_count = dict(sorted(emotion_count.items()))
    return sorted_emotion_count

# Creating function that saves the output to a CSV
def create_csv(data, output_path):
    # Using pandas dataframe and assigning column names
    emotion_table = pd.DataFrame(list(data.items()), columns=["Emotion", "Count"])
    # Making it into CSV with no index column
    emotion_table.to_csv(output_path, index=False)

# Creating function for plotting the emotion count
def plot_emotion_distribution(emotion_count, title, output_path):
    # Extracting keys (emotions) from the emotion_count dictionary
    emotions = list(emotion_count.keys())
    # Extracting values (counts) from the emotion_count dictionary
    count = list(emotion_count.values())

    # Creating a bar plot with emotions as x-axis and counts as y-axis
    plt.bar(emotions, count)
    # Setting the label for the x-axis
    plt.xlabel("Emotion")
    # Setting the label for the y-axis
    plt.ylabel("Count")
    # Naming the plot
    plt.title(title)
    # Saving the plot to the specified output path
    plt.savefig(output_path)
    # Closing the plot 
    plt.close()


def main():
    # Making correct path for the zipfile to unzip
    zip_file_path = os.path.join(".", "in", "archive.zip")
    target_folder = os.path.join(".", "in")
    # Changing the current working directory to the parent directory of the "in" folder (had to include this to make it work)
    os.chdir(os.path.join(".", "in", ".."))
    # Calling the function to unzip the file
    unzip_file(zip_file_path, target_folder)
    folder_path = os.path.join(".", "in", "fake_or_real_news.csv")
    # Using function above to create real and fake news data from their respective labels
    real_news_data = load_and_filter(folder_path, 'REAL')
    fake_news_data = load_and_filter(folder_path, 'FAKE')

    # Making objects containing the titles for all the data, real title and fake titles
    all_titles = pd.concat([real_news_data["title"], fake_news_data["title"]])
    real_titles = real_news_data["title"]
    fake_titles = fake_news_data["title"]

    # The following lines of code makes a emotion count for all titles, creates a CSV to the folder called "out" and finally creates a bar plot and saves it to the "figs" folder
    emotion_list_all = classify_emotions(all_titles)
    emotion_count_all = emotion_counter(emotion_list_all)
    output_path = os.path.join(".", "out", "All_headings.csv")
    create_csv(emotion_count_all, output_path)
    plot_emotion_distribution(emotion_count_all, "Emotion Distribution - All Headlines",
                              os.path.join(".", "figs", "All_headings.png"))

    # This code follows the same principle as the chunk above 
    emotion_list_real = classify_emotions(real_titles)
    emotion_count_real = emotion_counter(emotion_list_real)
    output_path = os.path.join(".", "out", "Real_headings.csv")
    create_csv(emotion_count_real, output_path)
    plot_emotion_distribution(emotion_count_real, "Emotion Distribution - Real News",
                              os.path.join(".", "figs", "Real_headings.png"))

    # This code follows the same principle as the chunk above 
    emotion_list_fake = classify_emotions(fake_titles)
    emotion_count_fake = emotion_counter(emotion_list_fake)
    output_path = os.path.join(".", "out", "Fake_headings.csv")
    create_csv(emotion_count_fake, output_path)
    plot_emotion_distribution(emotion_count_fake, "Emotion Distribution - Fake News",
                              os.path.join(".", "figs", "Fake_headings.png"))

if __name__ == "__main__":
    main()
