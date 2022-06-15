# some libraries and starter code
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import pandas as pd
import pickle

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
        tf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            smooth_idf=True,
            use_idf=True
        )
        cv = CountVectorizer(
            max_features=self.max_features,
            stop_words='english',
        )
        models = []
        for vec in [tf, cv]:
            vec.fit(fake)
            models.append(pd.DataFrame(vec.transform(quest).todense(), columns=vec.vocabulary_))
        return dict(zip(['TF-IDF', 'CountVector'], models))

    def __split(self, col):
        df = self.news[[col, 'fake']].copy()
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