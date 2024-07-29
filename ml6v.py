import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def load_data(filename):
    df = pd.read_csv(filename)
    df = pd.get_dummies(df)
    return df

def main():
    filename = "tennisdata.csv"
    df = load_data(filename)
    
    # Split dataset into features and target variable
    X = df.drop(df.columns[-1], axis=1).values
    y = df[df.columns[-1]].values
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Initialize the Gaussian Naive Bayes classifier
    model = GaussianNB()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the classifier is: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
