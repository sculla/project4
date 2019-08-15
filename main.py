import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from openTSNE import TSNE
import seaborn as sns



if __name__ == '__main__':
    books = [
        '/Users/sculla/PycharmProjects/project4/data/Dance Dance Dance_df.pkl']

    for idx, book in enumerate(books):
        df = pd.read_pickle(book)
        cv_tfidf = TfidfVectorizer(stop_words='english')
        X_tfidf = cv_tfidf.fit_transform(df['text']).toarray()
        df_tfidf = pd.DataFrame(X_tfidf, columns=cv_tfidf.get_feature_names())
    vectorizer = CountVectorizer(stop_words='english')
    doc_word = vectorizer.fit_transform(example)
    doc_word.shape
    lsa = TruncatedSVD(50)
    doc_topic = lsa.fit_transform(X_tfidf)
    lsa.explained_variance_ratio_
    topic_word = pd.DataFrame(lsa.components_.round(3),
                              columns=cv_tfidf.get_feature_names())
    Vt = pd.DataFrame(doc_topic.round(5),
                      index=df['text'],
                      )
    embedding = TSNE().fit(Vt)
    # sns.scatterplot(embedding)