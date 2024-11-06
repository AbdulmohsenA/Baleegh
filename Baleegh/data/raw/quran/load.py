import pandas as pd
import numpy as np
import re
import os


quran_df = pd.read_csv('quran/Arabic-Original.csv', names=['Surah', 'Ayah', 'Text'], sep="|")
endf = pd.read_csv('quran/en.yusufali.csv')
translations = pd.read_csv('quran/main_df.csv')


translations = translations.iloc[:, 4:7]
translations['Translation4'] = endf['Text']

#Fixing a problem in the first Translation
translations.loc[:7, "Translation1"] = translations.loc[:7, "Translation4"]

quran_df = pd.concat([quran_df, translations], axis=1)


def prepare_english(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove text between parentheses (Explanatory text which is not originally written in arabic)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove dashes and commas
    text = re.sub(r'[-,:;’‘\"\']+', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text

# Combine dashed verses to complete the sentences.
for index, row in quran_df.iterrows():
    
    if any(row[column].endswith("-") for column in ['Translation1', 'Translation2', 'Translation3', 'Translation4']):
        for column in ['Translation1', 'Translation2', 'Translation3', 'Translation4', 'Text']:
            quran_df.at[index + 1, column] = f"{quran_df.at[index, column]} {quran_df.at[index + 1, column]}"
            quran_df.at[index, column] = np.nan

quran_df = quran_df[quran_df['Ayah'] != 1]
quran_df = quran_df.dropna()
quran_df.loc[:, 'Translation1':'Translation4'] = quran_df.loc[:, 'Translation1':'Translation4'].applymap(prepare_english)
quran_df.reset_index(drop=True, inplace=True)

# quran_df.to_excel('data.xlsx')

print("Go to https://translate.google.com/?sl=en&tl=ar&op=docs and translate the file, then put it in google drive")

print("After setting the file as public, put the id below")

id = "1eLb_OS12_SGkLflsNAbR3av63TsYtipo"
translated_df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{id}/export?format=csv")

nan_indices = translated_df[translated_df.isna().any(axis=1)].index
for index in nan_indices:
    translated_df.at[index, ' الترجمة2'] = translated_df.at[index, ' الترجمة3']

data = pd.concat([quran_df['Text'], translated_df.iloc[:, 4:]], axis=1)

data = data.map(lambda text: re.sub(r'[^\u0621-\u064A\s]+', '', text))