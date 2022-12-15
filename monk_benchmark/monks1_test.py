import pandas as pd
import numpy as np

from train_val import multi_parameters, search_space_dict    


#perform one-hot-encoding tranformation

monks_1_train = pd.read_csv('monks-1.train', header=None, sep = '\s+', index_col=None,
                            names = ['class','a1','a2','a3','a4','a5','a6','Id'])

monks_1_test = pd.read_csv('monks-1.test', header=None, sep = '\s+', index_col=None,
                            names = ['class','a1','a2','a3','a4','a5','a6','Id'])


monks_1_train[['a1','a2','a3','a4','a5','a6']] = monks_1_train[['a1','a2','a3','a4','a5','a6']].astype(str)
monks_1_test[['a1','a2','a3','a4','a5','a6']] = monks_1_test[['a1','a2','a3','a4','a5','a6']].astype(str)
  
one_hot_monks_1_train = pd.get_dummies(monks_1_train, columns = ['a1','a2','a3','a4','a5','a6'])
one_hot_monks_1_test = pd.get_dummies(monks_1_test, columns = ['a1','a2','a3','a4','a5','a6'])

train_columns = [x for x in one_hot_monks_1_train.columns[2:]] + [x for x in one_hot_monks_1_train.columns[:2]]
test_columns = [x for x in one_hot_monks_1_test.columns[2:]] + [x for x in one_hot_monks_1_test.columns[:2]]

one_hot_monks_1_train = one_hot_monks_1_train.reindex(columns = train_columns)
one_hot_monks_1_test = one_hot_monks_1_test.reindex(columns = test_columns)

one_hot_monks_1_train = one_hot_monks_1_train.drop(['Id'], axis = 1)
one_hot_monks_1_test = one_hot_monks_1_test.drop(['Id'], axis = 1)

X_train = np.array(one_hot_monks_1_train)
X_test = np.array(one_hot_monks_1_test)


#define a search space for result evaluation

search_space = search_space_dict(layers_range=[1], units_range=[4], eta_range=[0.8,0.9],
                        alpha_range=[0.6,0.9], lambda_range=[0.01, 0.1],
                        num_targets=1)    

#test and plot different configurations, obtain the best one

multi_parameters(search_space, X_train, X_test, 17, 600, 50, 'binary_classification', 0.5)




        
        
                                    