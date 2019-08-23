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
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.classification import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from textwrap import wrap



class data():
    def __init__(self):
        self.Data = pd.DataFrame()
        self.Target = pd.DataFrame()
        self.Title = pd.DataFrame()
        self.Text = pd.DataFrame()

def test_two():
    books = [
        'data/Dance Dance Dance_df.pkl', #novel
        'data/Norwegian Wood_df.pkl', # novel
        'data/The Elephant Vanishes_df.pkl', # short stories
        'data/Wild Sheep Chase_df.pkl', # novel
        'data/Wind_Pinball_df.pkl'] # 2 novelas
    df_full = pd.DataFrame()
    for idx, book in enumerate(books):
        df_intermed = pd.read_pickle(book)
        print(df_intermed.shape, df_full.shape)
        df_full = pd.concat([df_full,df_intermed], axis=0)
    print(df_full.shape)
    #todo min doc and max doc for tfidf
    cv_tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = cv_tfidf.fit_transform(df_full['text']).toarray()
    df_full.reset_index(drop=True, inplace=True)

    # Reduction of features
    #TODO replace SVD with UMAP see: https://umap-learn.readthedocs.io/en/latest/benchmarking.html
    lsa = TruncatedSVD(50)
    doc_topic = lsa.fit_transform(X_tfidf)

    # Topic graph
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    doc_topic_tsne = pd.DataFrame(tsne.fit_transform(doc_topic))
    doc_topic_tsne.reset_index(drop=True, inplace=True)
    df1 = pd.concat([doc_topic_tsne, df_full[['translator','title']]], axis=1)
    fig = go.Figure()
    for translator, gr in df1.groupby('translator'):
        fig.add_trace(go.Scatter(x=gr[0], y=gr[1], name=translator, mode='markers',
                        hovertext= gr['title'], marker={'opacity':.1}))  #, marker=[opacity=.2]))  # [df_full[['title', 'year_published', 'ch_idx']]])

    plot(fig, filename='graphs/test_7.html')

    # predictions

    for n in np.linspace(30,70,10):
        lsa = TruncatedSVD(int(n))
        doc_topic = lsa.fit_transform(X_tfidf)
        df2 = pd.concat([pd.DataFrame(doc_topic), df_full[['translator', 'title']]], axis=1)
        test_title_list = []
        train_title_list = []
        for title, df_2 in df2.groupby('title'):
            if 'Elephant' in title:
                globals()[str(title).replace(' ', '_')] = df_2
                test_title_list.append(globals()[str(title).replace(' ', '_')])
            else:
                globals()[str(title).replace(' ', '_')] = df_2
                train_title_list.append(globals()[str(title).replace(' ', '_')])
        test_df = pd.concat(test_title_list, axis=0)
        train_df = pd.concat(train_title_list, axis=0)
        gbc = GradientBoostingClassifier(random_state=42, verbose=1)
        gbc.fit(train_df.drop(['translator','title'], axis=1),train_df['translator'])
        print(f'LSA with n: {n}')
        print(classification_report(test_df['translator'],
                              gbc.predict(test_df.drop(['translator','title'], axis=1))))

    # logit = LogisticRegressionCV(penalty='elasticnet', n_jobs=-1, random_state=42, verbose=1)
    # logit.fit()

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


def run_presentation(early=False, n=10):

    books = [
        'data/Dance Dance Dance_df.pkl',
        'data/Norwegian Wood_df.pkl',
        'data/The Elephant Vanishes_df.pkl',
        'data/Wild Sheep Chase_df.pkl'
    ]
    books_df = []
    for idx, book in enumerate(books):
        books_df.append(pd.read_pickle(book))
    df = pd.concat(books_df, axis=0)


    book_data = data()

    book_data.Data, book_data.Target = df['text'], df['translator']
    book_data.Text, book_data.Title = df['text'], df['title']
    print(book_data.Data.shape)
    print(book_data.Target.shape)

    tfidf = TfidfVectorizer(ngram_range=(1,4), norm='l2',use_idf=True, min_df=10, sublinear_tf=True) # ,max_df=0.5, max_features=10000)
    sgd = TruncatedSVD(int(n)) # (tol=1e-3, alpha=1e-06, max_iter=80, penalty='elasticnet')
    tsne = TSNE(n_components=2, perplexity=40,n_iter_without_progress=50, verbose=1)
    y = book_data.Target.values
    X = tfidf.fit_transform(book_data.Data.values)
    X_1 = sgd.fit_transform(X)
    if early:
        return X_1, y, book_data.Text, book_data.Title
    X_2 = tsne.fit_transform(X_1)
    return X_2, y, df['text'], df['title']


# def pretty_graph(x,y,text,title):
#     concat_list = [x,y,text,title]
#     full_df = pd.concat(concat_list,axis=0, names=[0,1,'translator','text','title'])
#     tr_set = []
#     te_set = []
#     for title in df.title.values:
#         if 'Elephant' in title:
#             te_set.append(full_df[full_df['title'] == title])
#         else:
#             tr_set.append(full_df[full_df['title'] == title])
#     te_df = pd.concat(te_set, axis=1)
#     tr_df = pd.concat(tr_set, axis=1)
#     # df_x = pd.DataFrame(x)
#     # df_y = pd.DataFrame(y, columns=['translator'])
#     # df1 = pd.concat([df_x, df_y], axis=1)
#     fig = go.Figure()
#     text_list = list(map(lambda x_line: '<br>'.join(wrap(x_line,50)),tr_df.text.values))
#     for translator, gr in tr_df.groupby('translator'):
#         fig.add_trace(go.Scatter(x=gr[0], y=gr[1], name=translator, mode='markers',
#                                  marker={'opacity': .3}, hovertext=text_list))
#     plot(fig, filename='graphs/test_12.html')

if __name__ == '__main__':
    # pretty_graph(*run_presentation())
    # pretty_graph(run_presentation())
    n_top = 60
    x, y, text, title = run_presentation(True, n=n_top)
    textdf = pd.DataFrame(text)
    titledf = pd.DataFrame(title)
    df_x = pd.DataFrame(x)
    df_y = pd.DataFrame(y, columns=['translator'])
    # title.reset_index(drop=True,inplace=True)
    # text.reset_index(drop=True, inplace=True)
    concat_list = [df_x,df_y,textdf,titledf]
    for df in concat_list:
        df.reset_index(drop=True, inplace=True)
    full_df = pd.concat(concat_list,axis=1)
    tr_set = []
    te_set = []
    full_df = full_df[full_df['title'] != np.nan]
    for title in full_df.title.values:
        if 'Elephant' in title:
            te_set.append(full_df[full_df['title'] == title])
        else:
            tr_set.append(full_df[full_df['title'] == title])
    te_df = pd.concat(te_set, axis=1)
    tr_df = pd.concat(tr_set, axis=1)
    x_tr = tr_df[[0,1]]
    y_tr = tr_df['translator']
    x_te = te_df[[0,1]]
    y_te = te_df['translator']
    #x_tr, x_te, y_tr, y_te = train_test_split(x,y, random_state=42, shuffle=True, train_size=.75, stratify=y )
    gbc = GradientBoostingClassifier(random_state=42, verbose=1, n_estimators=10000, max_depth=10, n_iter_no_change=500)
    gbc.fit(x_tr, y_tr)
    print(n_top)
    print(classification_report(y_te,gbc.predict(x_te)))

