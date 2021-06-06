import pickle
import pandas as pd
import numpy as np


possible_discounts = [0, 5, 10, 15, 20]



def takeGender(name: str):
    last_letter_of_name = name.split()[0][-1]
    return "k" if last_letter_of_name == "a" else "m"

def drop_view_before_buy(group):
    indices = []
    buys = group.loc[group['event_type'] == 1]
    
    
    bought_product_ids = set(buys['product_id'].unique())
    group = group[~((group['event_type'] == 0) & (group['product_id'].isin(bought_product_ids)))]
    return group

def preprocess(df):
    df["gender"] = df["name"].apply(takeGender)
    df['event_type'] = df['event_type'].map('BUY_PRODUCT': 1, 'VIEW_PRODUCT': 0)
    df['gender'] = df['gender'].map('m': 1, 'k': 0)
    
    df = df.groupby(by='session_id').apply(drop_view_before_buy)
    df = df.reset_index('session_id', drop=True)
    
    df['month'] = pd.DatetimeIndex(df['timestamp']).month
    df['weekday'] = pd.DatetimeIndex(df['timestamp']).weekday
    df['day'] = pd.DatetimeIndex(df['timestamp']).day
    df['hour'] = pd.DatetimeIndex(df['timestamp']).hour
    datetime_cols = ['month', 'weekday', 'day']
    df = df.drop('timestamp', axis=1)
    
    categories_path = df['category_path'].str.split(';', expand=True)
    category_cols = []

    for i, col in enumerate(categories_path.columns):
        col_name = 'Category' + str(i)
        df[col_name] = categories_path[col].fillna('Brak')
        category_cols.append(col_name)


    df = df.drop('category_path', axis=1)
    
    
    return df


def load():
    with open('bin/lr_clf.pkl', 'rb') as fp:
        model_b = pickle.dump(fp)
    
    with open('bin/xgb_clf.pkl', 'rb') as fp:
        model_a = pickle.load( fp)
    
    with open('bin/encoder.pkl', 'rb') as fp:
        encoder = pickle.load(fp)

    return model_a, model_b, encoder



def argmax(adict):
    return max(adict, key=adict.get)

def assign_discount(df, model_a, encoder, model_b=None, user_id=None, test_ab=False, hash_fun=hash):

    categorical_cols = ['Category0', 'Category1', 'Category2', 'Category3', 'city']
    non_categorical_cols = ['offered_discount' , 'price', 'gender'] + ['month', 'weekday', 'day']
    
    df = preprocess(df)
    
    encoded = pd.DataFrame(data=encoder.transform(df[categorical_cols]).toarray(),
                             columns=encoder.get_feature_names(categorical_cols), index=df.index, dtype=bool)*1
    X = pd.concat([df[non_categorical_cols], encoded], axis=1)
    

    
    if test_ab:
        is_group_a = hash_fun(user_id) % 2
        model = model_a if is_group_a==1 else model_b
    else:
        model = model_a
        
    
    profits = {}
    for discount in possible_discounts:
        X['offered_discount'] = discount
        y = model.predict(X)
        
        profit = sum(X['price'] * (1 - 0.01 * discount) * y)
        profits[discount] = profit
    
    return argmax(profits)
    
    
    

model_a, model_b, encoder = load()

while zgloszenia:
    assign_discount(zgloszenie_df, model_a, encoder, model_b, user_id=zgloszenie.uid, test_ab=True)
    

    
