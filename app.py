from tkinter import *
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
     	  
global filename
global df

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def preprocess():
    global df
    if df is not None:
        text.delete('1.0', END)
        text.insert(END, f"Dataset Shape: {df.shape}\n\n")
        text.insert(END, f"===============================================================================\n")
        text.insert(END, f"Missing Values:\n{df.isnull().sum()}\n\n")

def diseases_graph(): 
    global df
    if df is not None:
        # Get the count of each disease
        disease_counts = df['diseases'].value_counts()

        # Plot a pie chart with enhanced aesthetics
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette('pastel')[0:len(disease_counts)]
        plt.pie(disease_counts, labels=disease_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'black'})
        plt.title('Distribution of Diseases', fontsize=18, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def top10():
    global top_drugs
    top_drugs = df['ayurvedic_medicines'].value_counts().head(10)
    
    # Plot top 10 drugs by review count with enhanced aesthetics
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")
    sns.barplot(x=top_drugs.values, y=top_drugs.index, palette='magma')
    plt.xlabel('Review Count', fontsize=14, fontweight='bold')
    plt.ylabel('Drug Name', fontsize=14, fontweight='bold')
    plt.title('Top 10 Ayurvedic Medicines by Review Count', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for index, value in enumerate(top_drugs.values):
        plt.text(value, index, str(value), fontsize=12, fontweight='bold', color='black', va='center')
    plt.tight_layout()
    plt.show()

def Frequent_conditions():
    global top_conditions
    top_conditions = df['diseases'].value_counts().head(10)
    
    # Plot top 10 frequent conditions with enhanced aesthetics
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")
    sns.barplot(x=top_conditions.values, y=top_conditions.index, palette='coolwarm')
    plt.xlabel('Frequency', fontsize=14, fontweight='bold')
    plt.ylabel('Condition', fontsize=14, fontweight='bold')
    plt.title('Top 10 Frequent Diseases', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for index, value in enumerate(top_conditions.values):
        plt.text(value, index, str(value), fontsize=12, fontweight='bold', color='black', va='center')
    plt.tight_layout()
    plt.show()

def Model_Training():
    def submit_condition():
        user_condition = condition_entry.get()
        if user_condition:
            df_filtered = df[['ayurvedic_medicines==========>medicine_quantity', 'diseases']].dropna(subset=['diseases'])

            # Create TF-IDF matrix
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['diseases'])

            # Transform user input to TF-IDF vector
            user_condition_tfidf = tfidf_vectorizer.transform([user_condition])

            # Calculate cosine similarity
            similarity_scores = cosine_similarity(user_condition_tfidf, tfidf_matrix)

            # Get top recommended medicines
            top_indices = similarity_scores.argsort()[0][::-1][:3]  # Select top 3 indices
            top_medicines = df_filtered['ayurvedic_medicines==========>medicine_quantity'].iloc[top_indices]

            # Display top recommended medicines in the main text box
            text.delete('1.0', END)
            text.insert(END, f"Top recommended medicines for {user_condition}:\n")
            for medicine in top_medicines:
                text.insert(END, f"{medicine}\n")

    # Create a new window for input
    input_window = Toplevel(main)
    input_window.title("Enter Health Condition")
    input_window.geometry("400x200")
    input_window.config(bg="skyblue")

    # Create input label and entry widget
    condition_label = Label(input_window, text="Enter your health condition:")
    condition_label.pack(pady=10)
    condition_entry = Entry(input_window, width=50)
    condition_entry.pack(pady=10)

    # Create submit button
    submit_button = Button(input_window, text="Submit", command=submit_condition)
    submit_button.pack(pady=10)

# Main window
main = tk.Tk()
main.title("Exploring Ayurvedic Medicine Recommendation Using Machine Learning Techniques") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = tk.Label(main, text='Exploring Ayurvedic Medicine Recommendation Using Machine Learning Techniques', font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=15)
uploadButton.place(x=50, y=600)
uploadButton.config(font=font1)

preprocessButton = tk.Button(main, text="Preprocess Data", command=preprocess, bg="light yellow", width=15)
preprocessButton.place(x=250, y=600)
preprocessButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)     
pathlabel.place(x=450, y=600)

diseases_graph_button = tk.Button(main, text="Diseases Graph", command=diseases_graph, bg="light green", width=20)
diseases_graph_button.place(x=50, y=650)
diseases_graph_button.config(font=font1)

Frequent_conditions_button = tk.Button(main, text="Frequent Conditions", command=Frequent_conditions, bg="pink", width=20)
Frequent_conditions_button.place(x=250, y=650)
Frequent_conditions_button.config(font=font1)

top10_drogs = tk.Button(main, text="Top 10 Drugs", command=top10, bg="lightgrey", width=15)
top10_drogs.place(x=450, y=650)
top10_drogs.config(font=font1)

rec_1 = tk.Button(main, text="Model Training", command=Model_Training, bg="yellow", width=15)
rec_1.place(x=630, y=650)
rec_1.config(font=font1)

main.config(bg='#32d1a7')
main.mainloop()
