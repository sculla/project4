#!/anaconda3/envs/metis/bin/python3

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import re
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import matplotlib.pyplot as plt

def test_two():
    books = [
        'data/Dance Dance Dance_df.pkl',
        'data/Norwegian Wood_df.pkl',
        'data/The Elephant Vanishes_df.pkl',
        'data/Wild Sheep Chase_df.pkl',
        'data/Wind_Pinball_df.pkl']
    df_full = pd.DataFrame()
    for idx, book in enumerate(books):
        df_intermed = pd.read_pickle(book)
        print(df_intermed.shape, df_full.shape)
        df_full = pd.concat([df_full,df_intermed], axis=0)
    print(df_full.shape)
    cv_tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = cv_tfidf.fit_transform(df_full['text']).toarray()
    target = df_full['translator']

    return X_tfidf, target



def test_one():
    books = [
        '/Users/sculla/PycharmProjects/project4/data/Dance Dance Dance_df.pkl']

    for idx, book in enumerate(books):
        df = pd.read_pickle(book)
        cv_tfidf = TfidfVectorizer(max_features=5000)
        X_tfidf = cv_tfidf.fit_transform(df['text']).toarray()
        # df_tfidf = pd.DataFrame(X_tfidf, columns=cv_tfidf.get_feature_names())
    # vectorizer = CountVectorizer(stop_words='english')
    # doc_word = vectorizer.fit_transform(df['text'])
    # doc_word.shape
    lsa = TruncatedSVD(50)
    doc_topic = lsa.fit_transform(X_tfidf)
    # topic_word = pd.DataFrame(lsa.components_.round(3),
    #                           columns=cv_tfidf.get_feature_names())
    Vt = pd.DataFrame(doc_topic.round(5),
                      index=df['text'],
                      )
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    embedding = tsne.fit_transform(doc_topic)
    return embedding
    # sns.scatterplot(embedding)
    # plt.show()


def get_similar(nn, countvect, feature_names, word_vecs, query, n=10):

    if query in countvect.vocabulary_:
        query_index = countvect.vocabulary_[query]
        dist, index = nn.kneighbors(word_vecs[[query_index], :], n_neighbors=n)
        return ([(feature_names[i[0]], d[0]) for (d, i) in zip(dist.transpose(), index.transpose())])
    else:
        return "Query not in the dataset!"


def load_model(book_idx):
    with open(f'data/book{book_idx}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def main_word_proc(book_idx):
    """
    :param: book_idx Book Indexes:
                        Dance Dance Dance
                        Norwegian Wood
                        The Elephant Vanishes
                        Wild Sheep Chase
                        Wind / Pinball
    :return: NN trained on the words from the book
    """
    books = [
        'data/Dance Dance Dance_df.pkl',
        'data/Norwegian Wood_df.pkl',
        'data/The Elephant Vanishes_df.pkl',
        'data/Wild Sheep Chase_df.pkl',
        'data/Wind_Pinball_df.pkl']

    book = books[book_idx]
    ### concurrent processing of files
    #if os.path.exists(f'data/book{book_idx}_model.pkl'):
    #    continue
    ## Checkpoint
    #with open(f'data/book{book_idx}_model.pkl', 'wb') as f:
    #    pickle.dump(book,f)
    print(book)

    ## Pre-process
    df = pd.read_pickle(book)
    txt = []
    for text in df['text'].values:
        line = re.sub('\xa0', '', text)
        line = re.sub('’', '', line)

        txt.append(line)
    df['text'] = txt
    book_txt = '\n'.join(txt)

    ## CV Tokenizatoin
    sentences = sent_tokenize(book_txt)
    countvect = CountVectorizer(max_features=5000)
    bow = countvect.fit_transform(sentences)
    analyzer = countvect.build_analyzer()

    tfVec = TfidfVectorizer(max_features=5000)
    bow_tf = tfVec.fit_transform(sentences)
    analyzer_tf = tfVec.build_analyzer()

    ##
    context = []
    target = []
    for sentence in df['text']:
        line = analyzer(sentence)
        for idx in range(len(line) - 4):
            idx_f = idx + 2
            new_line = line[idx_f - 2:idx_f + 3]
            try:
                target.append(new_line.pop(2))
                context.append(' '.join(new_line))
            except IndexError:
                break

    ## Training MLP
    #X = countvect.transform(context)
    #y = np.array([countvect.vocabulary_.get(t, -1) for t in target])
    #model = MLPClassifier(hidden_layer_sizes=(30,), verbose=1, activation='identity')
    #model.fit(X, y)
    #try:
    #    f.close()
    #except:
    #    pass
    ## Checkpoint
    #with open(f'data/book{book_idx}_model.pkl', 'wb') as f:
    #    pickle.dump(model,f)
    model = load_model(book_idx)
    word_vecs = model.coefs_[0]
    feature_names = countvect.get_feature_names()
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    return nn.fit(word_vecs), countvect, feature_names, word_vecs


if __name__ == '__main__':
    df, target = test_two()
