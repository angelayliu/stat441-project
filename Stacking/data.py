import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Dataset():
    def __init__(self):
        X_train = pd.read_csv('../X_train.csv')
        
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        train_enc = ohe.fit_transform(X_train[['c_abrv', 'v228b', 'v231b', 'v233b', 'v251b', 'v275b_N2', 'v281a']])
        train_oh = pd.DataFrame(train_enc, columns=ohe.get_feature_names_out())
        X_train = pd.concat([X_train, train_oh], axis=1).reindex(X_train.index)
        
        X_train.drop('id', axis=1, inplace=True)
        # Drop string version column
        X_train.drop('c_abrv', axis=1, inplace=True)
        X_train.drop('v228b', axis=1, inplace=True)
        X_train.drop('v231b', axis=1, inplace=True)
        X_train.drop('v233b', axis=1, inplace=True)
        X_train.drop('v251b', axis=1, inplace=True)
        X_train.drop('v275b_N2', axis=1, inplace=True)
        X_train.drop('v275b_N1', axis=1, inplace=True)
        X_train.drop('v281a', axis=1, inplace=True)
        
        # Drop numerical version column
        X_train.drop('country', axis=1, inplace=True)
        X_train.drop('v228b_r', axis=1, inplace=True) #has missing
        X_train.drop('v231b_r', axis=1, inplace=True) #has missing
        X_train.drop('v233b_r', axis=1, inplace=True) #has missing
        X_train.drop('v251b_r', axis=1, inplace=True) #has missing
        X_train.drop('v275c_N2', axis=1, inplace=True)
        X_train.drop('v275c_N1', axis=1, inplace=True)
        X_train.drop('v281a_r', axis=1, inplace=True)

        self.X_train = X_train

        X_test = pd.read_csv('../X_test.csv')

        test_enc = ohe.transform(X_test[['c_abrv', 'v228b', 'v231b', 'v233b', 'v251b', 'v275b_N2', 'v281a']])
        test_oh = pd.DataFrame(test_enc, columns=ohe.get_feature_names_out())
        X_test = pd.concat([X_test, test_oh], axis=1).reindex(X_test.index)

        X_test.drop('id', axis=1, inplace=True)
        # Drop string version column
        X_test.drop('c_abrv', axis=1, inplace=True)
        X_test.drop('v228b', axis=1, inplace=True)
        X_test.drop('v231b', axis=1, inplace=True)
        X_test.drop('v233b', axis=1, inplace=True)
        X_test.drop('v251b', axis=1, inplace=True)
        X_test.drop('v275b_N2', axis=1, inplace=True)
        X_test.drop('v275b_N1', axis=1, inplace=True)
        X_test.drop('v281a', axis=1, inplace=True)
        
        # Drop numerical version column
        X_test.drop('country', axis=1, inplace=True)
        X_test.drop('v228b_r', axis=1, inplace=True) #has missing
        X_test.drop('v231b_r', axis=1, inplace=True) #has missing
        X_test.drop('v233b_r', axis=1, inplace=True) #has missing
        X_test.drop('v251b_r', axis=1, inplace=True) #has missing
        X_test.drop('v275c_N2', axis=1, inplace=True)
        X_test.drop('v275c_N1', axis=1, inplace=True)
        X_test.drop('v281a_r', axis=1, inplace=True)

        self.X_test = X_test

    def getTrain(self):
        return self.X_train

    def getTest(self):
        return self.X_test

