import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd



engine = create_engine('postgresql://dsi:correct horse battery staple@joshuacook.me')

def load_data_from_database():
    
    """loads dataset into pandas"""
    madelon_feat_df=pd.read_sql("SELECT * FROM Madelon", con=engine)

    return madelon_feat_df


def make_data_dict(X,y, random_state=None):
    
    """performs test_train_split on dataset"""
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=None)
    
    return {'X_train' : X_train,
    'X_test' : X_test,
    'y_train' : y_train,
    'y_test' : y_test}

def general_transformer(transformer,data_dict):
    
    """transforms data plugging in a transformer"""
    
    transformer.fit(data_dict['X_train'], data_dict['y_train'])
    
    data_dict['X_train']=transformer.transform(data_dict['X_train'])
    data_dict['X_test']=transformer.transform(data_dict['X_test'])
    
   
    
    return {'transformer':transformer,
            'X_train':data_dict['X_train'],
            'X_test':data_dict['X_test'],
            'y_train':data_dict['y_train'],
            'y_test': data_dict['y_test'],
            'data_dict':data_dict}

def general_model(model, data_dictionary):
    
    """makes it possible to pass model"""
    
    model.fit(data_dictionary['X_train'], data_dictionary['y_train'])
    
    
    data_dictionary['train_score'] = model.score(data_dictionary['X_train'], data_dictionary['y_train'])
    data_dictionary['test_score'] = model.score(data_dictionary['X_test'], data_dictionary['y_test'])
    
   
    
    return {'model':model,
            'train_score':data_dictionary['train_score'],
           'test_score': data_dictionary['test_score'],
            'X_train': data_dictionary['X_train'],
            'y_train':data_dictionary['y_train'],
            'X_test':data_dictionary['X_test'],
            'y_test':data_dictionary['y_test'],
           'data_dictionary': data_dictionary}