import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

""" Purpose of this script 
This script pre-processes the data scraped from the Palestine Chronicle website through the script 'palchr_scrape.py'
It is based on the tutorial found on https://github.com/robsalgado/personal_data_science_projects/blob/master/topic_modeling_nmf/nlp_topic_utils.ipynb
"""

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

# Contraction map
c_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}

# Compiling the contraction dict
c_re = re.compile('(%s)' % '|'.join(c_dict.keys()))

# List of stop words
add_stop = ['said', 'say', '...', 'like']
stop_words = ENGLISH_STOP_WORDS.union(add_stop)

# List of punctuation
punc = list(set(string.punctuation))

# Splits words on white spaces (leaves contractions intact) and splits out
# trailing punctuation
def casual_tokenizer(text):
    #tokenizer = TweetTokenizer()
    #tokens = tokenizer.tokenize(text)
    tokens = word_tokenize(text)            # I'm just using a normal word tokenizer, not the twitter one
    return tokens

def expandContractions(text, c_re=c_re):
    def replace(match):
        return c_dict[match.group(0)]
    return c_re.sub(replace, text)

def pos_tagging(text):      # added this myself for NER --> 'west' 'bank' should be one word...
    return pos_tag(text)

def process_text(text):
    text = casual_tokenizer(text)               # TOKENIZE
    # TODO: text = pos_tagging(text)                    # POS TAGGING: https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
    text = [each.lower() for each in text]      # TURN INTO LOWERCASE
    text = [re.sub('[0-9]+', '', each) for each in text]       # REMOVE DIGITS           # might be informative...
    text = [expandContractions(each, c_re=c_re) for each in text]   # EXPAND CONTRACTIONS
    text = [SnowballStemmer('english').stem(each) for each in text] # STEMMING
    text = [w for w in text if w not in punc]                       # REMOVE PUNCTUATION
    text = [w for w in text if w not in stop_words]                 # REMOVE STOP WORDS
    text = [each for each in text if len(each) > 1]                 # REMOVE SINGLE CHARACTERS
    text = [each for each in text if ' ' not in each]                # DUE TO CONTRACTIONS, REMOVE WORDS THAT HAVE EXTRA SPACES
    return text

def word_count(text):
    return len(str(text).split(' '))
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

df = pd.read_csv('PalestineChronicle_News.csv')
df = df[df['body'].notna()]

""" report: I'm not removing things like "By Palestine Chronicle Staff" because this will automatically be adjusted for
by TF-IDF -- appears in every article so algorithm won't think this is relevant anyway """

# --------------------------------------------------------------------------------------------------------
# TOKENIZATION, STEMMING, CONVERTING TO LOWERCASE, EXPANDING CONTRACTIONS
# REMOVING OF STOP WORDS, PUNCTUATION, DIGITS, SINGLE CHARACTERS

body_processed = [process_text(b) for b in df['body']]

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

""" What are the top 25 words used in the articles? """
all_words = sum(body_processed, [])
c = Counter(all_words)

# top 25 most common words
top_25 = pd.DataFrame(
    Counter(c).most_common(25),
    columns=['word', 'frequency']
)
# visualize top 25 most common words
fig = plt.figure(figsize=(25, 7))

g = sns.barplot(
    x='word',
    y='frequency',
    data=top_25,
    palette='GnBu_d'
)

g.set_xticklabels(
    g.get_xticklabels(),
    rotation=45,
    fontsize=14
)

plt.yticks(fontsize=14)
plt.xlabel('Words', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Top 25 Words', fontsize=17)

file_name = 'top_words'
file_path = 'images/'

fig.savefig(
    file_path + file_name + '.png',
    dpi=fig.dpi,
    bbox_inches='tight'
)

plt.show()


# Number of unique words after pre-processing:
print(f'Number of unique words after pre-processing: {len(set(all_words))}')    # result: 72687

# Compare to before pre-processing:
# remove punctuation and digits for more reasonable comparison
words_before = [casual_tokenizer(text) for text in df['body']]
words_before = [[w for w in text if w not in punc] for text in words_before]
all_words_before = sum(words_before, [])
print(f'Number of unique words before pre-processing: {len(set(all_words_before))}')    # result: 115820

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS FOR WORD COUNT PER ARTICLE

# Before pre-processing:
df['word_count_before'] = df['body'].apply(word_count)
print('Word count before pre-processing:')
df['word_count_before'].describe()

# TODO: Results

print(f'median   {df["word_count_before"].median()}') #median   1003.0
print('\n')

# After pre-processing:
df['word_count_after'] = [len(x) for x in body_processed]
print('Word count after pre-processing:')   # median   520.0
df['word_count_after'].describe()
print(f'median   {df["word_count_after"].median()}')

# TODO: Results

# Plot a hist of the word counts
for w in ['before', 'after']:
    fig = plt.figure(figsize=(10,5))

    plt.hist(
        df['word_count_' + w],
        bins=20,
        rwidth=0.9
    )

    plt.title('Distribution - Article Word Count, ' + w + ' pre-processing', fontsize=16)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Word Count', fontsize=12)
    plt.yticks(np.arange(0, 400, 50))
    plt.xticks(np.arange(0, 2000, 250))

    file_name = 'hist_word_count_' + w

    fig.savefig(
        file_path + file_name + '.png',
        dpi=fig.dpi,
        bbox_inches='tight'
    )

    plt.show()

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# finally, add body_processed to dataFrame
df['preprocessed_body'] = body_processed

df.to_csv('PalChr_preprocessed.csv', index=False)

