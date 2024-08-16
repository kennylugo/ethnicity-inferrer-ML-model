from preprocessing import run_preprocessing_logic
from training import run_training
from classification import classify_dna_sample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Central configuration for DNA files and models
dna_files = [
    # 'DNA.txt',
]

models = [
    ('DecisionTree', DecisionTreeClassifier(random_state=42)),
    # ('RandomForest', RandomForestClassifier(random_state=42)),
    # ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
    # ('AdaBoost', AdaBoostClassifier(random_state=42)),
    # ('ExtraTrees', ExtraTreesClassifier(random_state=42)),
    # ('LogisticRegression', LogisticRegression(max_iter=10000, random_state=42)),
    # ('SVM', SVC(probability=True, random_state=42)),
    # ('MLP', MLPClassifier(max_iter=10000, random_state=42)),
    # ('NaiveBayes', GaussianNB())
]

if __name__ == "__main__":
    # Load the preprocessing logic
    processed_genotype_data, feature_matrix, aims_data_df = run_preprocessing_logic()
    labels = processed_genotype_data.drop_duplicates('Sample')[['Sample', 'Ethnicity']].set_index('Sample')
    
    # Run the training and classification
    results = run_training(feature_matrix, labels, models)

    # Display results for each model
    for result in results:
        print(f"Model: {result[0]}, Accuracy: {result[1]:.4f}, Precision: {result[2]:.4f}, Recall: {result[3]:.4f}, F1 Score: {result[4]:.4f}")

    # Example of classifying a DNA sample with the best model found
    for model_name, accuracy, precision, recall, f1 in results:
        trained_model = models[[name for name, _ in models].index(model_name)][1]
        if trained_model:
            for dna_file_path in dna_files:
                snp_list = aims_data_df['Position'].tolist()
                ethnicity_labels = trained_model.classes_

                top_3_ethnicities = classify_dna_sample(dna_file_path, trained_model, snp_list, ethnicity_labels)
                print(f"Top 3 Predicted Ethnicities for {model_name} on {dna_file_path}: {top_3_ethnicities}")
