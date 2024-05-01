import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def get_processed_data():
    
    parentdir = "./propaganda_dataset_v2"
    train_file= "propaganda_train.tsv"
    val_file= "propaganda_val.tsv"

    train_path=os.path.join(parentdir,train_file)
    val_path=os.path.join(parentdir,val_file)
    train_df=pd.read_csv(train_path,delimiter="\t",quotechar='|')
    val_df=pd.read_csv(val_path,delimiter="\t",quotechar='|')

    merged = pd.concat([train_df, val_df], axis = 0)
    train_df, val_df = train_test_split(merged, test_size=0.3, shuffle=True,random_state=1) 
    val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=True, random_state=2)
    
    transformed_train_df = transform_df(train_df)
    transformed_val_df = transform_df(val_df)
    transformed_test_df = transform_df(test_df)
    
    return transformed_train_df, transformed_val_df, transformed_test_df




def show_class_distros(dataframe, data_type='unspecified'):
    labels = dataframe["snippet_label"].unique()
    label_counts = [len(dataframe[dataframe["snippet_label"]==label]) for label in labels]
    
    # Create a color palette with a unique color for each bar
    palette = sns.color_palette("hsv", len(labels))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))  # Adjust the figure size as needed
    fig.suptitle(f'{data_type} Data', fontsize=16)  # Add the figure title
    
    # Create the first barplot
    sns.barplot(x=labels, y=label_counts, palette=palette, ax=ax1)
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Label Counts')
    ax1.set_xticklabels(labels, rotation=90)  # Rotate x-axis labels by 90 degrees
    
    # Create the second barplot for propaganda vs non-propaganda
    non_propaganda_count = len(dataframe[dataframe["snippet_label"] == "not_propaganda"])
    propaganda_count = len(dataframe) - non_propaganda_count
    labels_prop = ["Non-Propaganda", "Propaganda"]
    counts_prop = [non_propaganda_count, propaganda_count]
    palette_prop = sns.color_palette("Set2", 2)  # Choose a different color palette
    sns.barplot(x=labels_prop, y=counts_prop, palette=palette_prop, ax=ax2)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Count')
    ax2.set_title('Propaganda vs Non-Propaganda Counts')
    
def get_snippet_from_sentence(single_sent):
  tokenized_sent = word_tokenize(single_sent)
  # print(tokenized_sent)
  start_idx = None
  end_idx = None
  for i, item in enumerate(tokenized_sent):
    if item == 'BOS':
      start_idx = i+2
    if item =='EOS':
      end_idx = i -1
  if start_idx is None or end_idx is None:
    raise ValueError("No BOS or EOS tags found in the sentence.")

  snippet = tokenized_sent[start_idx:end_idx]
  # print(snippet)
  # print(len(snippet))
  return snippet

def get_snippet_string_from_sentence(single_sent):
    pattern = r'<BOS>(.*?)<EOS>'
    matches = re.findall(pattern, single_sent)
    return matches[0].strip()
  

def remove_tags_from_sentence(single_sent):
    tokenized_sent = word_tokenize(single_sent)

    tokenized_sent = [tokenized_sent[i] for i in range(len(tokenized_sent)) if (i == 0 or tokenized_sent[i-1] != 'BOS') and tokenized_sent[i] != 'BOS' and (i == len(tokenized_sent)-1 or tokenized_sent[i+1] != 'BOS')]
    tokenized_sent = [tokenized_sent[i] for i in range(len(tokenized_sent)) if (i == 0 or tokenized_sent[i-1] != 'EOS') and tokenized_sent[i] != 'EOS' and (i == len(tokenized_sent)-1 or tokenized_sent[i+1] != 'EOS')]

    return tokenized_sent

def transform_strip_tag_and_tokenize(row):
    new_value = remove_tags_from_sentence(row['original'])
    return new_value
  
def transform_strip_tag(row):
    sent = row['original']
    cleaned_string = sent.replace("<BOS>", "")
    cleaned_string = cleaned_string.replace("<EOS>", "")
    return cleaned_string

def transform_extract_snippet_tokens(row):
    new_value = get_snippet_from_sentence(row['original'])
    return new_value

def transform_extract_snippet_string(row):
    new_value = get_snippet_string_from_sentence(row['original'])
    return new_value
  
def transform_binaryify(row):
    new_value = 0 if row['snippet_label'] == 'not_propaganda' else 1
    return new_value

def transform_df(dataframe):
    transformed_df = dataframe.copy()

    # Apply the transformation and create a new column
    transformed_df['original'] = transformed_df["tagged_in_context"]
    del transformed_df["tagged_in_context"]
    transformed_df['snippet_label'] = transformed_df["label"]
    del transformed_df["label"]
    transformed_df['sentence_tokenised_no_tags'] = transformed_df.apply(transform_strip_tag_and_tokenize, axis=1)
    transformed_df['original_sentence_no_tags'] = transformed_df.apply(transform_strip_tag, axis=1)
    transformed_df['snippet_tokenised'] = transformed_df.apply(transform_extract_snippet_tokens, axis=1)
    transformed_df['snippet_original'] = transformed_df.apply(transform_extract_snippet_string, axis=1)
    transformed_df['propaganda'] = transformed_df.apply(transform_binaryify, axis=1)
    
    new_column_order = ['propaganda', 'snippet_label', 'original', 'original_sentence_no_tags', 'snippet_original', 'sentence_tokenised_no_tags','snippet_tokenised']
    transformed_df = transformed_df[new_column_order]
    

    return transformed_df