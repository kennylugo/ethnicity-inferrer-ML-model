import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Function to train a model
def train(df, labels, model):
    model.fit(df, labels)
    return model

# Function to output training report
def output_training_report(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n")
    print(report)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Compute and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1

def run_training(feature_matrix, labels, models):
    y = labels.loc[feature_matrix.index, 'Ethnicity']
    results = []

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.3, random_state=42)

    # Train and test different models
    for model_name, model in models:
        print(f"Training model: {model_name}")
        trained_model = train(X_train, y_train, model)
        accuracy, precision, recall, f1 = output_training_report(trained_model, X_test, y_test)
        results.append((model_name, accuracy, precision, recall, f1))

    return results
