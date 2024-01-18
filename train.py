#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from joblib import dump

def load_and_preprocess_data(file_path):
    # Load the dataset
    laptop_data = pd.read_csv(file_path)

    # Preprocessing: Encoding categorical variables
    categorical_cols = laptop_data.select_dtypes(include=['object']).columns
    column_transformer = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

    # Splitting the data into features and target
    X = laptop_data.drop('Price_euros', axis=1)
    y = laptop_data['Price_euros']
    return column_transformer, X, y

def perform_grid_search(X_train, y_train):
    # Define a grid of hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

def main():
    file_path = 'S:/ML Course/Capstone 2/laptop_price.csv'
    column_transformer, X, y = load_and_preprocess_data(file_path)
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Applying the transformations
    X_train_transformed = column_transformer.fit_transform(X_train)
    X_test_transformed = column_transformer.transform(X_test)

    best_params, best_score = perform_grid_search(X_train_transformed, y_train)
    
    # Train and evaluate the model with the best parameters
    xgb_best = XGBRegressor(**best_params, random_state=42)
    xgb_best.fit(X_train_transformed, y_train)
    y_pred_xgb_best = xgb_best.predict(X_test_transformed)
    mse_xgb_best = mean_squared_error(y_test, y_pred_xgb_best)
    r2_xgb_best = r2_score(y_test, y_pred_xgb_best)

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    print("MSE:", mse_xgb_best)
    print("R2 Score:", r2_xgb_best)

    # Save the trained model to a file
    model_filename = 'xgb_best_model.joblib'
    dump(xgb_best, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()


# In[ ]:




