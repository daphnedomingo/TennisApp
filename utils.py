from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Model1Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []

    def fit(self, df, *args):
        dummy_columns=list(df.select_dtypes(include=['object']).columns)
        dummy_train =pd.get_dummies(data=df, columns=dummy_columns, drop_first=True)
        self.dummy_columns = dummy_columns
        self.dummy_train= dummy_train
        return self


    def transform(self, df, *args):
        dummy_columns=list(df.select_dtypes(include=['object']).columns)
        dummy_test = pd.get_dummies(data=df, columns=dummy_columns, drop_first=True)
        dummy_base= self.dummy_train
        dummy_base, dummy_test = dummy_base.align(dummy_test, join='left', axis=1)
        dummy_test.fillna(0,inplace=True)
        return dummy_test

class Model2Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []

    def get_player1(self,df):
        cols=['player2', '2_hand', '2_ht', '2_ioc', '2_age', '2_ace', '2_df', '2_1stSvPct',
          '2_1stSvWonPct', '2_bpfaced', '2_bpSavedPct', '2_rank']
        output_df= df.drop(labels=cols, axis=1)
        return output_df

    def fit(self, df, *args):
        df= self.get_player1(df)
        dummy_columns=list(df.select_dtypes(include=['object']).columns)
        dummy_train =pd.get_dummies(data=df, columns=dummy_columns, drop_first=True)
        self.dummy_columns = dummy_columns
        self.dummy_train= dummy_train
        return self


    def transform(self, df, *args):
        if len(df)>20:
            df=self.get_player1(df)
        dummy_test = pd.get_dummies(data=df, columns=self.dummy_columns, drop_first=True)
        dummy_base= self.dummy_train
        dummy_base, dummy_test = dummy_base.align(dummy_test, join='left', axis=1)
        dummy_test.fillna(0,inplace=True)
        return dummy_test


class Model3Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []

    def get_player2(self, df):
        cols=['player2', '2_hand', '2_ht', '2_ioc', '2_age', '2_ace', '2_df', '2_1stSvPct',
          '2_1stSvWonPct', '2_bpfaced', '2_bpSavedPct', '2_rank']
        cols2=['player1', '1_hand', '1_ht', '1_ioc', '1_age', '1_ace', '1_df', '1_1stSvPct',
          '1_1stSvWonPct', '1_bp_faced', '1_bpSavedPct', '1_rank']
        output_df= df.drop(labels=cols2, axis=1)
        new_cols=dict(zip(cols, cols2))
        output_df.rename(columns=new_cols, inplace=True)
        return output_df

    def get_player1(self,df):
        cols=['player2', '2_hand', '2_ht', '2_ioc', '2_age', '2_ace', '2_df', '2_1stSvPct',
          '2_1stSvWonPct', '2_bpfaced', '2_bpSavedPct', '2_rank']
        output_df= df.drop(labels=cols, axis=1)
        return output_df


    def fit(self, df, *args):
        base_df=df
        self.base_df= base_df
        df= self.get_player2(df)
        dummy_columns=list(df.select_dtypes(include=['object']).columns)
        dummy_train =pd.get_dummies(data=df, columns=dummy_columns, drop_first=True)
        self.dummy_columns = dummy_columns
        self.dummy_train= dummy_train
        return self

    def transform(self, df, *args):
        base_df=self.base_df
        if df.equals(base_df):
            dummy_test=self.dummy_train
        else:
            df=self.get_player1(df)
            dummy_test = pd.get_dummies(data=df, columns=self.dummy_columns, drop_first=True)
            dummy_base= self.dummy_train
            dummy_base, dummy_test = dummy_base.align(dummy_test, join='left', axis=1)
            dummy_test.fillna(0,inplace=True)
        return dummy_test


class Model4Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []

    def get_player2(self, df):
        cols=['player2', '2_hand', '2_ht', '2_ioc', '2_age', '2_ace', '2_df', '2_1stSvPct',
          '2_1stSvWonPct', '2_bpfaced', '2_bpSavedPct', '2_rank']
        cols2=['player1', '1_hand', '1_ht', '1_ioc', '1_age', '1_ace', '1_df', '1_1stSvPct',
          '1_1stSvWonPct', '1_bp_faced', '1_bpSavedPct', '1_rank']
        output_df= df.drop(labels=cols2, axis=1)
        new_cols=dict(zip(cols, cols2))
        output_df.rename(columns=new_cols, inplace=True)
        return output_df

    def get_player1(self,df):
        cols=['player2', '2_hand', '2_ht', '2_ioc', '2_age', '2_ace', '2_df', '2_1stSvPct',
          '2_1stSvWonPct', '2_bpfaced', '2_bpSavedPct', '2_rank']
        output_df= df.drop(labels=cols, axis=1)
        return output_df

    def get_inverse(self,df):
        cols=['player2', '2_hand', '2_ht', '2_ioc', '2_age', '2_ace', '2_df', '2_1stSvPct',
          '2_1stSvWonPct', '2_bpfaced', '2_bpSavedPct', '2_rank']
        cols2=['player1', '1_hand', '1_ht', '1_ioc', '1_age', '1_ace', '1_df', '1_1stSvPct',
          '1_1stSvWonPct', '1_bp_faced', '1_bpSavedPct', '1_rank']

        output_df= self.get_player2(df)
        add_df= df[cols2]
        add_df.columns= cols
        output_df= output_df.merge(add_df, left_index=True, right_index=True)
        return output_df


    def fit(self, df, *args):
        base_df=df
        self.base_df= base_df
        df= self.get_inverse(df)
        dummy_columns=list(df.select_dtypes(include=['object']).columns)
        dummy_train =pd.get_dummies(data=df, columns=dummy_columns, drop_first=True)
        self.dummy_columns = dummy_columns
        self.dummy_train= dummy_train
        return self

    def transform(self, df, *args):
        base_df=self.base_df
        if df.equals(base_df):
            dummy_test=self.dummy_train
        else:
            dummy_test = pd.get_dummies(data=df, columns=self.dummy_columns, drop_first=True)
            dummy_base= self.dummy_train
            dummy_base, dummy_test = dummy_base.align(dummy_test, join='left', axis=1)
            dummy_test.fillna(0,inplace=True)
        return dummy_test
