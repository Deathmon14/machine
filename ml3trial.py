import numpy as np
import pandas as pd

def learn(concepts, target):
    '''
    learn() function implements the learning method of the Candidate Elimination algorithm.

    Arguments:
        concepts - a 2D array with all the features
        target - a 1D array with corresponding output values

    Returns:
        specific_h - the final specific hypothesis
        general_h - the final general hypothesis
    '''
    # Initialize specific_h with the first instance from concepts
    specific_h = concepts[0].copy()

    # Initialize general_h to represent the most general hypothesis
    general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    # The learning iterations
    for i, h in enumerate(concepts):
        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        # Checking if the hypothesis has a negative target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hypothesis, change values only in G
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("\nSteps of Candidate Elimination Algorithm", i+1)
        print("Specific Hypothesis:", specific_h)
        print("General Hypothesis:", general_h)

    # Remove empty rows from general_h
    general_h = [h for h in general_h if h != ['?'] * len(specific_h)]

    # Return final values
    return specific_h, general_h

# Loading Data from a CSV File
try:
    data = pd.read_csv('output.csv')
except FileNotFoundError:
    print("Error: File not found.")
    exit(1)

# Handling Categorical Data
# Assuming the last column is the target
target_col = data.columns[-1]
data[target_col] = data[target_col].astype(str)

# Separating concept features from Target
concepts = data.iloc[:, :-1].values

# Isolating target into a separate array
target = data.iloc[:, -1].values

# Learn from the data
s_final, g_final = learn(concepts, target)

print("\nFinal Specific_h:", s_final, sep="\n")
print("\nFinal General_h:", g_final, sep="\n")
