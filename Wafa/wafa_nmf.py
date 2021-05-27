import pandas as pd
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from operator import itemgetter
from ast import literal_eval

""" Purpose of this script 
This script performs topic modeling using NMF
It is based on the tutorial found on https://github.com/robsalgado/personal_data_science_projects/blob/master/topic_modeling_nmf/nlp_topic_utils.ipynb
"""

num_topics = None   # set to None if we want to let coherence scores decide the best number of topics

# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def top_words(topic, n_top_words):
    return topic.argsort()[:-n_top_words - 1:-1]

def topic_table(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        t = (topic_idx)
        topics[t] = [feature_names[i] for i in top_words(topic, n_top_words)]
    return pd.DataFrame(topics)

def whitespace_tokenizer(text):
    pattern = r"(?u)\b\w\w+\b"
    tokenizer_regex = RegexpTokenizer(pattern)
    tokens = tokenizer_regex.tokenize(text)
    return tokens

# Funtion to remove duplicate words
def unique_words(text):
    ulist = []
    [ulist.append(x) for x in text if x not in ulist]
    return ulist

#-----------------------------------------------------------------------

df = pd.read_csv('Wafa_preprocessed.csv')
df['preprocessed_body'] = df['preprocessed_body'].apply(literal_eval)
file_path = 'NMF/images'    # to store charts
#-----------------------------------------------------------------------

texts = df['preprocessed_body']

# Create a dictionary: a mapping between words and their integer id
dictionary = Dictionary(texts)

""" e.g. dictionary.token2id:
 'khadija': 695,
 'shaher': 696,
 'threaten': 697,
 'tighten': 698,
 'administr': 699,

 dictionary.cfs     collection frequencies: how many instances of token are contained in the docs
 dictionary.dfs     document frequencies: how many docs contain this token
"""

print('\n')
print('Gensim dictionary summaries to mention in report')
print(f'Num docs processed: {dictionary.num_docs}')
print(f'Num words processed: {dictionary.num_pos}')
print(f'Num non-zeroes in BOW matrix (sum of the number of unique words per doc over the entire corpus:'
      f' {dictionary.num_nnz}')

# Filter out tokens in the dictionary by their frequency to limit the number of features
dictionary.filter_extremes(
    no_below=3,  # Keep tokens which are contained in at least no_below documents.
    no_above=0.85,
    # Keep tokens which are contained in no more than no_above documents (fraction of total corpus size, not an absolute number).
    # afraid it will filter out Israel & Palestine... (originally set to 0.85) --> add in keep_tokens
    keep_n=5000,  # Keep only the first keep_n most frequent tokens.
    # keep_tokens=['isra', 'palestinian', 'israel']
)

# Create the bag-of-words format (list of (token_id, token_count) for each doc)
corpus = [dictionary.doc2bow(text) for text in texts]

# Create a list of the topic numbers we want to try
if num_topics is None:
    topic_nums = list(np.arange(5, 75 + 1, 5))  # TODO: way too much?

    # Run the nmf model and calculate the coherence score
    # for each number of topics
    coherence_scores = []

    for num in topic_nums:
        # TODO: look into parameters
        nmf = Nmf(
            corpus=corpus,
            num_topics=num,
            id2word=dictionary,
            chunksize=2000,
            passes=5,
            kappa=.1,
            minimum_probability=0.01,
            w_max_iter=300,
            w_stop_condition=0.0001,
            h_max_iter=100,
            h_stop_condition=0.001,
            eval_every=10,
            normalize=True,
            random_state=42
        )

        # Run the coherence model to get the score
        cm = CoherenceModel(
            model=nmf,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'  # TODO: what is this?
        )

        coherence_scores.append(round(cm.get_coherence(), 5))

    # Get the number of topics with the highest coherence score
    scores = list(zip(topic_nums, coherence_scores))
    num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

    # Visualize
    fig = plt.figure(figsize=(15, 7))

    plt.plot(
        topic_nums,
        coherence_scores,
        linewidth=3,
        color='#4287f5'
    )

    plt.xlabel("Topic Num", fontsize=14)
    plt.ylabel("Coherence Score", fontsize=14)
    plt.title('Coherence Score by Topic Number - Best Number of Topics: {}'.format(num_topics), fontsize=18)
    plt.xticks(np.arange(5, max(topic_nums) + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)

    file_name = 'c_score'

    fig.savefig(
        file_path + file_name + '.png',
        dpi=fig.dpi,
        bbox_inches='tight'
    )

    plt.show()

    # Use the number of topics with the
    # highest coherence score to run the
    # sklearn nmf model

# Create the tfidf weights
tfidf_vectorizer = TfidfVectorizer(
    min_df=3,
    max_df=0.85,
    max_features=5000,
    ngram_range=(1, 2),
    preprocessor=' '.join
)

tfidf = tfidf_vectorizer.fit_transform(texts)

# Save the feature names for later to create topic summaries
tfidf_fn = tfidf_vectorizer.get_feature_names()

# Run the nmf model
nmf = NMF(
    n_components=num_topics,
    init='nndsvd',
    max_iter=500,
    l1_ratio=0.0,
    solver='cd',
    alpha=0.0,
    tol=1e-4,
    random_state=42
).fit(tfidf)

# Use the top words for each cluster by tfidf weight
# to create 'topics'

# Getting a df with each topic by document
docweights = nmf.transform(tfidf_vectorizer.transform(texts))

n_top_words = 8

topic_df = topic_table(
    nmf,
    tfidf_fn,
    n_top_words
).T

# Cleaning up the top words to create topic summaries
topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1)  # Joining each word into a list
topic_df['topics'] = topic_df['topics'].str[0]  # Removing the list brackets
topic_df['topics'] = topic_df['topics'].apply(lambda x: whitespace_tokenizer(x))  # tokenize
topic_df['topics'] = topic_df['topics'].apply(lambda x: unique_words(x))  # Removing duplicate words
topic_df['topics'] = topic_df['topics'].apply(lambda x: [' '.join(x)])  # Joining each word into a list
topic_df['topics'] = topic_df['topics'].str[0]  # Removing the list brackets

topic_df.head()

# Create a df with only the created topics and topic num
topic_df = topic_df['topics'].reset_index()
topic_df.columns = ['topic_num', 'topics']

topic_df.head()

topic_df.to_csv('NMF/topics/NMF_' + str(num_topics), index=False)

# ---------------------

""" Interesting for NMF 25 topics:
topic 11: netanyahu protest demonstr charg prime minist corrupt
"""
