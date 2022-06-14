# some libraries and starter code
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass

@dataclass
class FakeNews:
    """
    Fake News Aggregator:
    ------------------------------------------
    - Clean and transform fake news data
    """
    __base_fold = __file__.rsplit('/', 1)[0]

    def __post_init__(self):
        self.news = self.__get_news()
        self.headline_bag = self.__clean('title')
        self.text_bag = self.__clean('text')

    @staticmethod
    def __tfidf(series_data, max_features=25):
        vec = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            smooth_idf=True,
            use_idf=True
        )
        vec.fit(series_data)
        return vec

    def __clean(self, col):
        df = self.news[[col, 'fake']].copy()
        fake = df[df.fake == 'F'][col]
        questioned = df[df.fake == 'Q'][col]
        vr = self.__tfidf(fake)
        return pd.DataFrame(vr.transform(questioned).todense(), columns=vr.vocabulary_)

    def __get_news(self):

        data_fold = os.path.join(self.__base_fold, 'data')
        df_lst = []
        for fname in os.listdir(data_fold):
            fpath = os.path.join(data_fold, fname)
            df = pd.read_excel(fpath, header=None).iloc[:, 1::2]
            df_lst.append(df)
        full_df = pd.concat(df_lst)
        full_df.columns = ['date', 'title', 'text', 'url', 'medium']
        full_df['fake'] = 'F'
        full_df.loc[full_df.medium == 'hinnews.com', 'fake'] = 'Q'

        return full_df[['title', 'text', 'fake']].dropna().reset_index(drop=True)

if __name__ == '__main__':
    test = FakeNews()