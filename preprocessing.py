import os
import pandas as pd
import pysam
from loguru import logger
from tabulate import tabulate

# Define file paths as constants
VCF_DIR = "/Users/kennybatista/Downloads/VCF/"
METADATA_FILE = "/Users/kennybatista/Downloads/VCF/igsr-1000 genomes phase 3 release.tsv"
BED_FILE_PATH = "246_COMBINED_AISNPS.BED"

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
def run_preprocessing_logic():
    sample_metadata_dict = load_sample_metadata(METADATA_FILE)
    aims_data_df = parse_aims_file(BED_FILE_PATH)
    genotype_data = fetch_genotype_data(VCF_DIR, aims_data_df, sample_metadata_dict)
    feature_matrix = process_genotype_data(genotype_data, aims_data_df['Position'])
    return genotype_data, feature_matrix, aims_data_df
