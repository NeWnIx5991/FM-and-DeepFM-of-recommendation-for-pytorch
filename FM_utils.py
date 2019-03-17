import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def utils():
    all_dataset = pd.read_csv('data/adult_data.csv')
    print('features num initially : ',len(all_dataset.columns))


    '''
    ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'gender',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
           'income_bracket', 'income_label']
    '''

    age_list = (np.arange(11) * 10).tolist()
    temp_list = ((np.arange(10)+1) * 10).tolist()
    label_list = []

    for i,j in zip(age_list,temp_list):
        label_list.append(str(i) + '-' + str(j))

    all_dataset['age'] = pd.cut(all_dataset['age'],age_list,labels = label_list)
    all_dataset['hours_per_week'] = pd.cut(all_dataset['hours_per_week'],age_list,labels = label_list)

    label_dataset = all_dataset['income_label']

    all_dataset.drop(['fnlwgt','education_num','capital_gain','capital_loss','income_bracket','income_label'],axis = 1,inplace=True)
    print('features num after drop : ',len(all_dataset.columns))

    print(all_dataset.shape)


    all_data_arr = pd.get_dummies(all_dataset)
    print(all_data_arr.shape[1])

    col_names = all_dataset.columns.values.tolist()

    for col in col_names:
        col_unique_index = dict()
        col_unique_name = all_dataset[col].unique()
        for index,unique_name in enumerate(col_unique_name):
            col_unique_index[unique_name] = index
        all_dataset[col] = [col_unique_index[item] for item in all_dataset[col]]


    seed = 2019
    X_train , X_test = train_test_split(all_data_arr.values,test_size=0.3,random_state=seed)
    Y_train , Y_test = train_test_split(np.array(label_dataset),test_size=0.3 , random_state=seed)



    features_num = []
    for col in col_names:
        if col == 'age':
            features_num.append(10)
        else:
            features_num.append(len(all_dataset[col].unique()))

    print(sum(features_num))
    return X_train,Y_train,X_test,Y_test,features_num