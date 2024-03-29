import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Dataset():
    def __init__(self):
        nominal_cols = ['country', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24',
                        'v20a', 'v20b', 
                        'v24a_IT', 'v24b_IT',
                        'v25', 'v26', 'v27', 'v28', 'v29', 'v30',
                        'v30a', 'v30b', 'v30c',
                        'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 
                        'v45a', 'v45b', 'v45c',
                        'v51', 'v52', 'v53', 
                        'v52_cs',
                        'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 
                        'v71',
                        'v72_DE', 'v73_DE', 'v74_DE', 'v75_DE', 'v76_DE', 'v77_DE', 'v78_DE', 'v79_DE',
                        'v85', 'v86', 'v87', 'v88', 'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 
                        'v96a', 'v96b',
                        'v108', 'v109', 'v110', 'v111', 'v112', 'v113', 'v114',
                        'v169',
                        'v174_cs', 'v175_cs',
                        'v204',
                        'v225',
                        'v227',
                        'v228b',
                        'v230',
                        'v231b',
                        'v232',
                        'v233b',
                        'v234', 'v235', 'v236', 'v237', 'v238',
                        'v244', 'v245',
                        'v246_egp',
                        'v248', 'v249', 'v250',
                        'v251b',
                        'v253', 'v254',
                        'v255_egp',
                        'v257',
                        'v259', 'v260',
                        'v264', 'v265',
                        'v275c_N2', 'v275c_N1',
                        'v281a_r',
                        'v282',
                        'f20',
                        'f24_IT',
                        'f30a',
                        'f45a',
                        'f46_IT',
                        'f85',
                        'f96',
                        'f108',
                        'f110',
                        'f112_SE',
                        'f252_edulvlb_CH']
        
        # has missing: 'v228b_r', 'v231b_r', 'v233b_r', 'v251b_r'
        cols_to_drop = ['id', 'c_abrv', 'v228b_r', 'v231b_r', 'v233b_r', 'v251b_r', 'v275b_N2', 'v275b_N1', 'v281a',
                        'v243_edulvlb', 'v243_edulvlb_2', 'v243_edulvlb_1', 'v243_ISCED_3', 'v243_ISCED_2', 'v243_ISCED_2b', 'v243_ISCED_1', 'v243_EISCED', 'v243_ISCED97',
                        'v243_cs', 'v243_cs_DE1', 'v243_cs_DE2', 'v243_cs_DE3', 'v243_cs_GB1', 'v243_cs_GB2', 
                        'v246_ISCO_2', 'v246_SIOPS', 'v246_ISEI', 'v246_ESeC',
                        'v252_edulvlb', 'v252_edulvlb_2', 'v252_edulvlb_1', 'v252_ISCED_3', 'v252_ISCED_2', 'v252_ISCED_2b', 'v252_ISCED_1', 'v252_EISCED', 'v252_ISCED97',
                        'v252_cs', 'v252_cs_DE1', 'v252_cs_DE2', 'v252_cs_DE3', 'v252_cs_GB1', 'v252_cs_GB2',
                        'v255_ISCO_2', 'v255_SIOPS', 'v255_ISEI', 'v255_ESeC',
                        'v262_edulvlb', 'v262_edulvlb_2', 'v262_edulvlb_1', 'v262_ISCED_3', 'v262_ISCED_2', 'v262_ISCED_2b', 'v262_ISCED_1', 'v262_EISCED', 'v262_ISCED97',
                        'v262_cs', 'v262_cs_DE1', 'v262_cs_DE2', 'v262_cs_DE3', 'v262_cs_GB1', 'v262_cs_GB2',
                        'v263_edulvlb', 'v263_edulvlb_2', 'v263_edulvlb_1', 'v263_ISCED_3', 'v263_ISCED_2', 'v263_ISCED_2b', 'v263_ISCED_1', 'v263_EISCED', 'v263_ISCED97',
                        'v263_cs', 'v263_cs_DE1', 'v263_cs_DE2', 'v263_cs_DE3', 'v263_cs_GB1', 'v263_cs_GB2',
                        'age', 'v241', 'v242']
        

        X_train = pd.read_csv('../X_train.csv')
        X_train.drop(cols_to_drop, axis=1, inplace=True)
        
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train[nominal_cols] = X_train[nominal_cols].astype(str)
        train_enc = ohe.fit_transform(X_train[nominal_cols])
        train_oh = pd.DataFrame(train_enc, columns=ohe.get_feature_names_out())
        X_train = pd.concat([X_train, train_oh], axis=1).reindex(X_train.index)
        
        X_train.drop(nominal_cols, axis=1, inplace=True)
        X_train[X_train <= -2] = np.nan

        sev_to_four = ['v171', 'v172', 'v173']
        sixtysix_to_zero = ['v243_8cat', 'v243_r', 
                            'v252_8cat', 'v252_r',
                            'v262_r','v263_r',
                            'v266']
        sixthou_to_zero = ['v262_8cat', 'v263_8cat']
        six_to_zero = ['v267', 'v268', 'v269', 'v270', 'v271', 'v272', 'v273', 'v274']
        X_train[sev_to_four] = X_train[sev_to_four].replace(7, 4)
        X_train[sixtysix_to_zero] = X_train[sixtysix_to_zero].replace(66, 0)
        X_train[sixthou_to_zero] = X_train[sixthou_to_zero].replace(6666, 0)
        X_train[six_to_zero] = X_train[six_to_zero].replace(6, 0)

        self.X_train = X_train


        X_test = pd.read_csv('../X_test.csv')
        X_test.drop(cols_to_drop, axis=1, inplace=True)

        X_test[nominal_cols] = X_test[nominal_cols].astype(str)
        test_enc = ohe.transform(X_test[nominal_cols])
        test_oh = pd.DataFrame(test_enc, columns=ohe.get_feature_names_out())
        X_test = pd.concat([X_test, test_oh], axis=1).reindex(X_test.index)

        X_test.drop(nominal_cols, axis=1, inplace=True)
        X_test[X_test <= -2] = np.nan

        X_test[sev_to_four] = X_test[sev_to_four].replace(7, 4)
        X_test[sixtysix_to_zero] = X_test[sixtysix_to_zero].replace(66, 0)
        X_test[sixthou_to_zero] = X_test[sixthou_to_zero].replace(6666, 0)
        X_test[six_to_zero] = X_test[six_to_zero].replace(6, 0)

        self.X_test = X_test

    def getTrain(self):
        return self.X_train

    def getTest(self):
        return self.X_test