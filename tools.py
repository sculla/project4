from tika import parser
from bs4 import BeautifulSoup as bs
import pandas as pd
import pickle
import numpy as np

with open('data/columns.pkl', 'rb') as f:
    columns = pickle.load(f)
del f

pdf_paths = [
    '/Users/sculla/Calibre Library/Haruki Murakami/Dance Dance Dance (Vintage International) (16)/Dance Dance Dance (Vintage International) - Haruki Murakami.pdf',
    '/Users/sculla/Calibre Library/Haruki Murakami/Norwegian Wood (Vintage International) (15)/Norwegian Wood (Vintage International) - Haruki Murakami.pdf',
    '/Users/sculla/Calibre Library/Haruki Murakami/A Wild Sheep Chase_ A Novel (Trilogy of the Rat Book 3) (14)/A Wild Sheep Chase_ A Novel (Trilogy of th - Haruki Murakami.pdf',
    '/Users/sculla/Calibre Library/Haruki Murakami/The Elephant Vanishes_ Stories (Vintage International) (13)/The Elephant Vanishes_ Stories (Vintage In - Haruki Murakami.pdf',
    '/Users/sculla/Calibre Library/Haruki Murakami/Wind_Pinball_ Two novels (12)/Wind_Pinball_ Two novels - Haruki Murakami.pdf'
]

txt_paths = [
    '/Users/sculla/Calibre Library/Haruki Murakami/Dance Dance Dance (Vintage International) (16)/Dance Dance Dance (Vintage International) - Haruki Murakami.txt',
    '/Users/sculla/Calibre Library/Haruki Murakami/Norwegian Wood (Vintage International) (15)/Norwegian Wood (Vintage International) - Haruki Murakami.txt',
    '/Users/sculla/Calibre Library/Haruki Murakami/A Wild Sheep Chase_ A Novel (Trilogy of the Rat Book 3) (14)/A Wild Sheep Chase_ A Novel (Trilogy of th - Haruki Murakami.txt',
    '/Users/sculla/Calibre Library/Haruki Murakami/The Elephant Vanishes_ Stories (Vintage International) (13)/The Elephant Vanishes_ Stories (Vintage In - Haruki Murakami.txt',
    '/Users/sculla/Calibre Library/Haruki Murakami/Wind_Pinball_ Two novels (12)/Wind_Pinball_ Two novels - Haruki Murakami.txt'

]

def combo_cosign():
    books = [
        'data/Dance Dance Dance_df.pkl',
        'data/Norwegian Wood_df.pkl',
        'data/The Elephant Vanishes_df.pkl',
        'data/Wild Sheep Chase_df.pkl',
        'data/Wind_Pinball_df.pkl']
    for idx, book in enumerate(books):
        df = pd.read_pickle(book)
        cv_tfidf = TfidfVectorizer(stop_words='english')
        X_tfidf = cv_tfidf.fit_transform(df['text']).toarray()
        df_tfidf = pd.DataFrame(X_tfidf, columns=cv_tfidf.get_feature_names())


        pairs = list(combinations(enumerate(df['text']), 2))
        combos = [(a[0], b[0]) for a, b in pairs]
        phrases = [(a[1], b[1]) for a, b in pairs]
        results = [
            cosine_similarity(np.array(df_tfidf.iloc[a]).reshape(1, -1), np.array(df_tfidf.iloc[b]).reshape(1, -1)) \
            for a, b in combos
        ]

        with open(f'data/book_2-{idx}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print('whaaaa')

def load_pdf():
    for book in pdf_paths:
        yield parser.from_file(book)


def load_txt():
    for book in txt_paths:
        with open(book, 'r') as f:
            yield f.readlines()


def para_replace(book_text):
    for idx, val in enumerate(book_text[:-1]):
        if (book_text[idx + 1] == '\n') & (val == '\n'):
            book_text[idx] = '---'

    new_book_text = []
    for idx, val in enumerate(book_text):
        if not (book_text[idx - 1] == '---') & (val == '---' or val == '\n'):
            new_book_text.append(val)
    del book_text

    text_out = []
    for idx, val in enumerate(new_book_text):
        if not (val == '\n'):
            text_out.append(val)
    return text_out


def load_EV():
    book = pd.DataFrame(columns=columns)
    title = 'The Elephant Vanishes'
    published = 1987
    translated = 2000
    # ev = 3 - 19
    EV_titles = ['THE WIND-UP BIRD AND TUESDAY’S WOMEN',
                 'THE SECOND BAKERY ATTACK',
                 'THE KANGAROO COMMUNIQUÉ',
                 'ON SEEING THE 100% PERFECT GIRL ONE BEAUTIFUL APRIL MORNING',
                 'SLEEP',
                 'THE FALL OF THE ROMAN EMPIRE, THE 1881 INDIAN UPRISING, HITLER’S INVASION OF POLAND, AND THE REALM OF RAGING WINDS',
                 'LEDERHOSEN',
                 'BARN BURNING',
                 'THE LITTLE GREEN MONSTER',
                 'FAMILY AFFAIR',
                 'A WINDOW',
                 'TV PEOPLE',
                 'A SLOW BOAT TO CHINA',
                 'THE DANCING DWARF',
                 'THE LAST LAWN OF THE AFTERNOON',
                 'THE SILENCE',
                 'THE ELEPHANT VANISHES']
    for ch_idx in range(17):
        part_num = str(ch_idx + 3).zfill(2)
        with open(f'data/The Elephant Vanishes/text/part00{part_num}.html', 'r') as f:
            soup = bs(f, 'html.parser')
        containers = soup.find_all('p')
        translator = ' '.join(containers[-1].text.split()[2:])
        for cont_idx, container in enumerate(containers):
            row = [f'{title}_{EV_titles[ch_idx]}',
                   container.text,
                   published,
                   translator,
                   translated,
                   ch_idx,
                   cont_idx
                   ]
            book = book.append(dict(zip(columns, row)), ignore_index=True)
    book.to_pickle(f'data/{title}_df.pkl')
    return book


def load_DDD():
    book = pd.DataFrame(columns=columns)
    # ddd = 004 start 47 end
    title = 'Dance Dance Dance'
    translator = 'Alfred Birnbaum'
    published = 1988
    translated = 1994
    for ch_idx in range(4, 48):
        part_num = str(ch_idx).zfill(2)
        with open(f'data/Dance Dance Dance/text/part00{part_num}.html', 'r') as f:
            soup = bs(f, 'html.parser')
        containers = soup.find_all('p')
        for cont_idx, container in enumerate(containers):
            row = [f'{title}',
                   container.text,
                   published,
                   translator,
                   translated,
                   ch_idx,
                   cont_idx
                   ]
            book = book.append(dict(zip(columns, row)), ignore_index=True)
    book.to_pickle(f'data/{title}_df.pkl')
    return book


def load_NW():
    book = pd.DataFrame(columns=columns)
    title = 'Norwegian Wood'
    translator = 'Jay Rubin'
    published = 1987
    translated = 2000
    # NW = 3-13
    for ch_idx in range(3, 14):
        part_num = str(ch_idx).zfill(2)
        with open(f'data/Norwegian Wood/text/part00{part_num}.html', 'r') as f:
            soup = bs(f, 'html.parser')
        containers = soup.find_all('p')
        for cont_idx, container in enumerate(containers):
            row = [f'{title}',
                   container.text,
                   published,
                   translator,
                   translated,
                   ch_idx,
                   cont_idx
                   ]
            book = book.append(dict(zip(columns, row)), ignore_index=True)
    book.to_pickle(f'data/{title}_df.pkl')
    return book


def load_WP():
    book = pd.DataFrame(columns=columns)
    title = 'Wind_Pinball'
    translator = 'Ted Goossen'
    published = 1980
    translated = 2015
    # w = 7 p = 9
    for ch_idx in [7, 9]:
        part_num = str(ch_idx).zfill(2)
        with open(f'data/Wind_Pinball/text/part00{part_num}.html', 'r') as f:
            soup = bs(f, 'html.parser')
        containers = soup.find_all('p')
        for cont_idx, container in enumerate(containers):
            row = [f'{title}',
                   container.text,
                   published,
                   translator,
                   translated,
                   ch_idx,
                   cont_idx
                   ]
            book = book.append(dict(zip(columns, row)), ignore_index=True)
    book.to_pickle(f'data/{title}_df.pkl')
    return book


def load_WSC():
    book = pd.DataFrame(columns=columns)
    title = 'Wild Sheep Chase'
    translator = 'Alfred Birnbaum'
    published = 1982
    translated = 1989
    # WSC = 3 - 54, skip 3,5,8,12,19,24,33,38
    for ch_idx in range(3, 55):
        if ch_idx in [3, 5, 8, 12, 19, 24, 33, 38]:
            continue
        part_num = str(ch_idx).zfill(2)
        with open(f'data/A Wild Sheep Chase/text/part00{part_num}.html', 'r') as f:
            soup = bs(f, 'html.parser')
        containers = soup.find_all('p')
        for cont_idx, container in enumerate(containers):
            row = [f'{title}',
                   container.text,
                   published,
                   translator,
                   translated,
                   ch_idx,
                   cont_idx
                   ]
            book = book.append(dict(zip(columns, row)), ignore_index=True)
    book.to_pickle(f'data/{title}_df.pkl')
    return book


# if __name__ == '__main__':
#     load_DDD()
#     load_EV()
#     load_NW()
#     load_WP()
#     load_WSC()
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import pickle

    df = pd.read_pickle('/Users/sculla/PycharmProjects/project4/data/Dance Dance Dance_df.pkl')
    from sklearn.feature_extraction.text import TfidfVectorizer

    cv_tfidf = TfidfVectorizer()
    X_tfidf = cv_tfidf.fit_transform(df['text']).toarray()
    df_tfidf = pd.DataFrame(X_tfidf, columns=cv_tfidf.get_feature_names())
    from sklearn.metrics.pairwise import cosine_similarity
    from itertools import combinations

    pairs = list(combinations(enumerate(df['text']), 2))
    combos = [(a[0], b[0]) for a, b in pairs]
    phrases = [(a[1], b[1]) for a, b in pairs]
    results = [
        cosine_similarity([np.array(df_tfidf.iloc[a])], [np.array(df_tfidf.iloc[b])])[0][0] \
        for a,b in combos
    ]

    with open('/Users/sculla/PycharmProjects/project4/data/test.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('whaaaa')
