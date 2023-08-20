

import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Utility Functions

def load_data(file_path):
    return pd.read_csv(file_path)

def convert_to_datetime(data, column):
    dummy = data.copy()
    dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
    return dummy

def convert_timestamp_to_hourly(data, column):
    dummy = data.copy()
    new_ts = dummy[column].tolist()
    new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
    new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
    dummy[column] = new_ts
    return dummy

def aggregate_dataframes(df, group_by_cols, agg_dict):
    return df.groupby(group_by_cols).agg(agg_dict).reset_index()

def merge_dataframes(df1, df2, on_cols, how='left'):
    return df1.merge(df2, on=on_cols, how=how)

# Modeling Functions

def prepare_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_and_evaluate(X, y, K=10, split=0.75):
    accuracy = []
    model = RandomForestRegressor()
    scaler = StandardScaler()

    for fold in range(0, K):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        trained_model = model.fit(X_train, y_train)
        y_pred = trained_model.predict(X_test)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
    return trained_model, sum(accuracy) / len(accuracy)

# Visualization Functions

def plot_feature_importance(model, X):
    features = [i.split("__")[0] for i in X.columns]
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 20))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# Main Function

def run_pipeline(sales_path, stock_path, temp_path):
    # Load data
    sales_df = load_data(sales_path)
    stock_df = load_data(stock_path)
    temp_df = load_data(temp_path)
    
    # Convert to datetime and hourly timestamp
    sales_df = convert_to_datetime(sales_df, 'timestamp')
    stock_df = convert_to_datetime(stock_df, 'timestamp')
    temp_df = convert_to_datetime(temp_df, 'timestamp')
    
    sales_df = convert_timestamp_to_hourly(sales_df, 'timestamp')
    stock_df = convert_timestamp_to_hourly(stock_df, 'timestamp')
    temp_df = convert_timestamp_to_hourly(temp_df, 'timestamp')
    
    # Aggregate dataframes
    sales_agg = aggregate_dataframes(sales_df, ['timestamp', 'product_id'], {'quantity': 'sum'})
    stock_agg = aggregate_dataframes(stock_df, ['timestamp', 'product_id'], {'estimated_stock_pct': 'mean'})
    temp_agg = aggregate_dataframes(temp_df, ['timestamp'], {'temperature': 'mean'})
    
    # Merge dataframes
    merged_df = merge_dataframes(stock_agg, sales_agg, ['timestamp', 'product_id'])
    merged_df = merge_dataframes(merged_df, temp_agg, 'timestamp')
    
    # Prepare data for modeling
    X, y = prepare_data(merged_df, 'estimated_stock_pct')
    
    # Train and evaluate model
    model, avg_mae = train_and_evaluate(X, y)
    
    # Plot feature importance
    plot_feature_importance(model, X)
    
    return avg_mae

