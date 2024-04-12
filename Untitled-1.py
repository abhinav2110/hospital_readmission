

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import joblib  # Import joblib for saving the model

# Load your dataset
df = pd.read_csv("hospital_readmissions.csv")

# Define columns for label encoding and custom mapping
ordinal_columns = ['age', 'glucose_test', 'A1Ctest']
binary_columns = ['change', 'diabetes_med', 'readmitted']

# Apply Label Encoding for ordinal columns
ordinal_encoder = LabelEncoder()
for col in ordinal_columns:
    df[col] = ordinal_encoder.fit_transform(df[col])

# Apply custom mapping for binary columns
binary_mapping = {'yes': 1, 'no': 0}
for col in binary_columns:
    df[col] = df[col].map(binary_mapping)

# Target encoding for specified columns
target_columns = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']
for col in target_columns:
    target_encoding = df.groupby(col)['readmitted'].mean()
    df[col] = df[col].map(target_encoding)

# Separate features and target variable
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
model_filename = "model.pkl"
joblib.dump(model, model_filename)

print("Model saved successfully as:", model_filename)

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load your trained model
load_model = joblib.load('model.pkl')

# Function to preprocess user input
def preprocess_user_input(user_input):
    # Define label encoder for ordinal columns
    ordinal_columns = ['age', 'glucose_test', 'A1Ctest','medical_specialty', 'diag_1', 'diag_2', 'diag_3']

    # Apply Label Encoding for ordinal columns
    ordinal_encoder = LabelEncoder()
    for col in ordinal_columns:
        if col in user_input:
            user_input[col] = ordinal_encoder.fit_transform([user_input[col]])[0]
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_user_input(user_input):
    # Define label encoder for ordinal columns
    ordinal_columns = ['age', 'glucose_test', 'A1Ctest', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']

    # Apply Label Encoding for ordinal columns
    ordinal_encoder = LabelEncoder()
    for col in ordinal_columns:
        if col in user_input:
            user_input[col] = ordinal_encoder.fit_transform([user_input[col]])[0]

    # Apply custom binary mapping to binary columns
    binary_mapping = {'yes': 1, 'no': 0}
    binary_columns = ['change', 'diabetes_med']
    for col in binary_columns:
        if col in user_input:
            user_input[col] = binary_mapping[user_input[col]]

    # Convert the preprocessed user input back to a pandas Series
    processed_input = pd.Series(user_input)

    return processed_input


# Function to predict readmission based on preprocessed user input
def predict_readmission(user_input):
    # Preprocess user input
    processed_input = preprocess_user_input(user_input)

    # Ensure processed_input is a Series or 1-dimensional array
    input_array = processed_input.values.reshape(1, -1)

    # Debugging: Print the shape and content of input_array
    # print("Shape of input_array:", input_array.shape)
    # print("Content of input_array:", input_array)

    # Make prediction using the trained model
    prediction =load_model.predict(input_array)

    return prediction[0]

# Example usage:
# user_input = {
#     'age': '50-60',
#     'time_in_hospital': 8,
#     'n_lab_procedures': 72,
#     'n_procedures': 1,
#     'n_medications': 18,
#     'n_outpatient': 0,
#     'n_inpatient': 0,
#     'n_emergency': 1,
#     'medical_specialty': 'Circulatory',
#     'diag_1': 'Circulatory',
#     'diag_2': 'Respiratory',
#     'diag_3': 'Other',
#     'glucose_test': 'normal',
#     'A1Ctest': 'no',
#     'change': 'no',
#     'diabetes_med': 'yes'
# }

# predicted_readmission = predict_readmission(user_input)
# print("Predicted Readmission:", predicted_readmission)