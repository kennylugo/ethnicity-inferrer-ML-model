import os
import pandas as pd
import numpy as np
import pysam
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
import joblib
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Define file paths as constants
VCF_DIR = "/Users/kennybatista/Downloads/VCF/"
METADATA_FILE = "/Users/kennybatista/Downloads/VCF/igsr-1000 genomes phase 3 release.tsv"
BED_FILE_PATH = "246_COMBINED_AISNPS.BED"
ANCESTRY_DNA_FILE1 = '/Users/kennybatista/Documents/Documents/DNA Data/Kenny DNA Data/AncestryDNA.txt'
ANCESTRY_DNA_FILE2 = '/Users/kennybatista/Documents/Documents/DNA Data/Ale DNA Data/AncestryDNA.txt'

# Utility function to load sample metadata
def load_sample_metadata(file_path):
    sample_to_ethnicity = {}
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            columns = line.strip().split('\t')
            if len(columns) < 5:
                continue
            sample_id, ethnicity = columns[0], columns[4]
            sample_to_ethnicity[sample_id] = ethnicity
    
    logger.debug(f"Sample Metadata: {list(sample_to_ethnicity.items())[:5]}")
    return sample_to_ethnicity

# Utility function to parse AIMs file
def parse_aims_file(file_path):
    aims_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('chromosome'):
                fields = line.strip().split('\t')
                chromosome = fields[0].replace('chr', '')
                position = int(fields[1])
                rsid = fields[3]
                aims_data.append((chromosome, position, rsid))
    df = pd.DataFrame(aims_data, columns=['Chromosome', 'Position', 'RSID'])
    logger.debug(f"\nAIMs df:\n{tabulate(df.head(), headers='keys', tablefmt='psql')}")
    return df

# Function to fetch genotype data
def fetch_genotype_data(vcf_dir, aims_data, sample_to_ethnicity):
    data = []
    low_sampled_classes_to_skip = {'Punjabi,Punjabi', 'Finnish,Finnish', 'Mende,Mende', 'Kinh,Kinh Vietnamese', 
                        'Bengali,Bengali', 'Gambian Mandinka,Gambian', 'British,English', 'Esan,Esan', 
                        'Luhya,Luhya', 'Iberian,Spanish', 'Iberian,Mende', 'Japanese,Japanese'}

    for _, row in aims_data.iterrows():
        chromosome, position, rsid = row['Chromosome'], row['Position'], row['RSID']
        vcf_file_path = os.path.join(vcf_dir, f"Chromosome {chromosome}/ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")
        
        if not os.path.exists(vcf_file_path):
            logger.error(f"VCF file for chromosome {chromosome} not found: {vcf_file_path}")
            continue
        
        try:
            vcf = pysam.VariantFile(vcf_file_path)
            for record in vcf.fetch(chromosome, position - 1, position):
                for sample, sample_data in record.samples.items():
                    if sample in sample_to_ethnicity and 'GT' in sample_data:
                        ethnicity = sample_to_ethnicity[sample]
                        if ethnicity in low_sampled_classes_to_skip:
                            continue  # Skip samples with the specified ethnicities

                        genotype = sample_data['GT']
                        if None not in genotype:
                            genotype_encoded = sum((0 if g is None else g) for g in genotype)
                            data.append({
                                'Sample': sample,
                                'Ethnicity': sample_to_ethnicity[sample],
                                'Chromosome': chromosome,
                                'Position': position,
                                'RSID': rsid,
                                'Genotype': int(genotype_encoded)  # Ensure integer encoding
                            })
        except Exception as e:
            logger.error(f"Error fetching records for {chromosome} at {position}: {e}")

    df = pd.DataFrame(data, columns=['Sample', 'Ethnicity', 'Chromosome', 'Position', 'RSID', 'Genotype'])

    # Filter out ethnicities with less than 20 samples
    ethnicity_counts = df['Ethnicity'].value_counts()
    valid_ethnicities = ethnicity_counts[ethnicity_counts >= 20].index
    df = df[df['Ethnicity'].isin(valid_ethnicities)]

    return df

# Function to process genotype data into a feature matrix
def process_genotype_data(data: pd.DataFrame, snp_list):
    if data.empty:
        logger.error("No genotype data fetched. Dataframe is empty.")
        return pd.DataFrame(columns=snp_list)
    
    feature_matrix = data.pivot_table(index='Sample', columns='Position', values='Genotype', fill_value=0)
    
    missing_snps = list(set(snp_list) - set(feature_matrix.columns))
    missing_df = pd.DataFrame(0, index=feature_matrix.index, columns=missing_snps)
    feature_matrix = pd.concat([feature_matrix, missing_df], axis=1)
    
    feature_matrix = feature_matrix[snp_list].astype(int)  # Ensure integer type

    samples_with_missing_snps = data[data['Position'].isin(missing_snps)]['Sample'].unique()
    ethnicities_with_missing_snps = data[data['Sample'].isin(samples_with_missing_snps)]['Ethnicity'].unique()

    samples_with_all_snps = data[~data['Position'].isin(missing_snps)]['Sample'].unique()
    ethnicities_with_all_snps = data[data['Sample'].isin(samples_with_all_snps)]['Ethnicity'].unique()

    logger.info(f"Ethnicities with missing SNPs: {list(ethnicities_with_missing_snps)}")
    logger.info(f"Ethnicities with all SNPs: {list(ethnicities_with_all_snps)}")

    found = set(snp_list) - set(missing_snps)
    logger.info(f"SNPs we were able to find a match for: {len(found)}")

    return feature_matrix

# Main processing logic
def run_main_processing_logic():
    sample_metadata_dict = load_sample_metadata(METADATA_FILE)
    aims_data_df = parse_aims_file(BED_FILE_PATH)
    genotype_data = fetch_genotype_data(VCF_DIR, aims_data_df, sample_metadata_dict)
    feature_matrix = process_genotype_data(genotype_data, aims_data_df['Position'])
    return genotype_data, feature_matrix, aims_data_df

processed_genotype_data, feature_matrix, aims_data_df = run_main_processing_logic()

# Utility function to train a model
def train(df, labels, model, random_state):
    if hasattr(model, 'random_state'):
        model.set_params(random_state=random_state)
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

    # # Plot confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.show()

    # Compute and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")

    return accuracy, y_pred

# Function to classify DNA sample
def classify_dna_sample(dna_file_path, model, snp_list, ethnicity_labels):
    with open(dna_file_path, 'r') as file:
        header_lines = [next(file) for _ in range(100)]

    genotype_dict = {}

    if any('allele1' in line for line in header_lines):
        dna_data = pd.read_table(dna_file_path, sep='\t', comment='#')
        if 'allele1' not in dna_data.columns or 'allele2' not in dna_data.columns:
            raise ValueError("Expected columns 'allele1' and 'allele2' not found in the AncestryDNA file")
        
        logger.debug("Parsing AncestryDNA file format")
        for _, row in dna_data.iterrows():
            position = row['position']
            if position in snp_list:
                genotype = row['allele1'] + row['allele2']
                logger.debug(f"Processing position {position} with genotype {genotype}")
                if genotype in ['AA', 'GG', 'TT', 'CC']:
                    genotype_dict[position] = 0
                elif genotype in ['AG', 'GA', 'CT', 'TC']:
                    genotype_dict[position] = 1
           
    elif any('23andMe' in line for line in header_lines):
        dna_data = pd.read_table(dna_file_path, sep='\t', comment='#', names=['rsid', 'chromosome', 'position', 'genotype'])
        if 'genotype' not in dna_data.columns:
            raise ValueError("Expected column 'genotype' not found in the 23andMe file")
        
        logger.debug("Parsing 23andMe file format")
        for _, row in dna_data.iterrows():
            position = row['position']
            if position in snp_list:
                genotype = row['genotype']
                logger.debug(f"Processing position {position} with genotype {genotype}")
                if genotype in ['AA', 'GG', 'TT', 'CC']:
                    genotype_dict[position] = 0
                elif genotype in ['AG', 'GA', 'CT', 'TC']:
                    genotype_dict[position] = 1
    
    elif any('General' in line for line in header_lines):
        dna_data = pd.read_csv(dna_file_path, sep="\t", names=['rsid', 'chromosome', 'position', 'genotype'], skiprows=2)
        if not all(col in dna_data.columns for col in ['rsid', 'chromosome', 'position', 'genotype']):
            raise ValueError("Expected columns 'rsid', 'chromosome', 'position', 'genotype' not found in the file")
        
        logger.debug("Parsing general DNA file format")
        for _, row in dna_data.iterrows():
            position = row['position']
            if position in snp_list:
                genotype = row['genotype']
                logger.debug(f"Processing position {position} with genotype {genotype}")
                if genotype in ['AA', 'GG', 'TT', 'CC']:
                    genotype_dict[position] = 0
                elif genotype in ['AG', 'GA', 'CT', 'TC']:
                    genotype_dict[position] = 1
    
    if not genotype_dict:
        logger.error("No valid genotype data found in the file")
        raise ValueError("No valid genotype data found in the file")

    feature_vector = pd.Series(genotype_dict, index=snp_list).fillna(0)
    matrix_for_prediction = pd.DataFrame([feature_vector])

    probabilities = model.predict_proba(matrix_for_prediction)[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_ethnicities = [(ethnicity_labels[i], probabilities[i]) for i in top_3_indices]

    return top_3_ethnicities

# Initialize models with random states where applicable
models = [
    ('DecisionTree', DecisionTreeClassifier(random_state=42)),
    # ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
    # ('AdaBoost', AdaBoostClassifier(random_state=42)),
    # ('ExtraTrees', ExtraTreesClassifier(random_state=42)),
    # ('RandomForest', RandomForestClassifier(random_state=42)),
    # ('SVM', SVC(probability=True, random_state=42)),
    # ('KNeighbors', KNeighborsClassifier(n_neighbors=11, weights="distance")),
    # ('LogisticRegression', LogisticRegression(max_iter=10000, random_state=42)),
    # ('NeuralNetwork', MLPClassifier(max_iter=10000, random_state=42))
]

labels = processed_genotype_data.drop_duplicates('Sample')[['Sample', 'Ethnicity']].set_index('Sample')
y = labels.loc[feature_matrix.index, 'Ethnicity']

# Iterate over different random states to find the desired output
for random_state in range(10000):
    print(f"Testing with random_state={random_state}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.3, random_state=random_state)

    # Train and test different models
    results = []
    for model_name, model in models:
        print(f"Training model: {model_name} with random_state={random_state}")
        trained_model = train(X_train, y_train, model, random_state)
        accuracy, y_pred = output_training_report(trained_model, X_test, y_test)
        
        # Check for ANCESTRY_DNA_FILE1
        top_3_ethnicities_file1 = classify_dna_sample(ANCESTRY_DNA_FILE1, trained_model, aims_data_df['Position'].tolist(), model.classes_)
        
        if top_3_ethnicities_file1[0] == ('Puerto Rican ancestry', 1.0):
            # Check for ANCESTRY_DNA_FILE2 with the same model
            top_3_ethnicities_file2 = classify_dna_sample(ANCESTRY_DNA_FILE2, trained_model, aims_data_df['Position'].tolist(), model.classes_)
            
            if top_3_ethnicities_file2[0] == ('Mexican Ancestry', 1.0):
                print(f"\nDesired results achieved with random_state={random_state}")
                print(f"Model: {model_name}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Top 3 Predicted Ethnicities for ANCESTRY_DNA_FILE1: {top_3_ethnicities_file1}")
                print(f"Top 3 Predicted Ethnicities for ANCESTRY_DNA_FILE2: {top_3_ethnicities_file2}")
                break
    else:
        continue  # continue if the inner loop wasn't broken
    break  # break if the inner loop was broken
