# some libraries and starter code
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import pandas as pd

@dataclass
class FakeNews:
    """
    Fake News Aggregator:
    ------------------------------------------
    - Clean and transform fake news data
    """
    __base_fold = __file__.rsplit('/', 1)[0]
    max_features = 25

    def __post_init__(self):
        self.__news = self.__get_news()
        self.models = {
            'headline': self.__run_models('title'),
            'body': self.__run_models('text')
        }

    def __run_models(self, col):
        fake, quest = self.__split(col)
        return {
            'TF-IDF': self.__get_tfidf(fake, quest),
            'CV': self.__get_cv(fake, quest)
        }

    def __get_tfidf(self, fake, quest):
        tf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            smooth_idf=True,
            use_idf=True
        )
        tf.fit(fake)
        return {
            'quest_df': pd.DataFrame(tf.transform(quest).todense(), columns=tf.vocabulary_),
            'idf': pd.Series(tf.idf_, tf.vocabulary_),
            'vectorizer': tf
        }

    def __get_cv(self, fake, quest):
        cv = CountVectorizer(
            max_features=self.max_features,
            stop_words='english',
        )
        cv.fit(fake)
        return {
            'quest_df': pd.DataFrame(cv.transform(quest).todense(), columns=cv.vocabulary_),
            'vectorizer': cv
        }

    def __split(self, col):
        df = self.__news[[col, 'fake']].copy()
        return df[df.fake == 'F'][col], df[df.fake == 'Q'][col]

    def __join_fold(self, fold):
        return os.path.join(self.__base_fold, fold)

    def __get_news(self):

        data_fold = self.__join_fold('data')
        df_lst = []
        for fname in os.listdir(data_fold):
            fpath = os.path.join(data_fold, fname)
            df = pd.read_excel(fpath, header=None, engine='openpyxl').iloc[:, 1::2]
            df_lst.append(df)
        full_df = pd.concat(df_lst)
        full_df.columns = ['date', 'title', 'text', 'url', 'medium']
        full_df['fake'] = 'F'
        full_df.loc[full_df.medium == 'hinnews.com', 'fake'] = 'Q'

        return full_df[['title', 'text', 'fake']].dropna().reset_index(drop=True)

if __name__ == '__main__':
    test = FakeNews()