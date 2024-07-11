import pandas as pd

# Function to classify DNA sample
def classify_dna_sample(dna_file_path, model, snp_list, ethnicity_labels):
    with open(dna_file_path, 'r') as file:
        header_lines = [next(file) for _ in range(100)]

    genotype_dict = {}

    if any('allele1' in line for line in header_lines):
        dna_data = pd.read_table(dna_file_path, sep='\t', comment='#')
        if 'allele1' not in dna_data.columns or 'allele2' not in dna_data.columns:
            raise ValueError("Expected columns 'allele1' and 'allele2' not found in the AncestryDNA file")
        
        print("Parsing AncestryDNA file format")
        for _, row in dna_data.iterrows():
            position = row['position']
            if position in snp_list:
                genotype = row['allele1'] + row['allele2']
                print(f"Processing position {position} with genotype {genotype}")
                if genotype in ['AA', 'GG', 'TT', 'CC']:
                    genotype_dict[position] = 0
                elif genotype in ['AG', 'GA', 'CT', 'TC']:
                    genotype_dict[position] = 1
           
    elif any('23andMe' in line for line in header_lines):
        dna_data = pd.read_table(dna_file_path, sep='\t', comment='#', names=['rsid', 'chromosome', 'position', 'genotype'])
        if 'genotype' not in dna_data.columns:
            raise ValueError("Expected column 'genotype' not found in the 23andMe file")
        
        print("Parsing 23andMe file format")
        for _, row in dna_data.iterrows():
            position = row['position']
            if position in snp_list:
                genotype = row['genotype']
                print(f"Processing position {position} with genotype {genotype}")
                if genotype in ['AA', 'GG', 'TT', 'CC']:
                    genotype_dict[position] = 0
                elif genotype in ['AG', 'GA', 'CT', 'TC']:
                    genotype_dict[position] = 1
    
    elif any('General' in line for line in header_lines):
        dna_data = pd.read_csv(dna_file_path, sep="\t", names=['rsid', 'chromosome', 'position', 'genotype'], skiprows=2)
        if not all(col in dna_data.columns for col in ['rsid', 'chromosome', 'position', 'genotype']):
            raise ValueError("Expected columns 'rsid', 'chromosome', 'position', 'genotype' not found in the file")
        
        print("Parsing general DNA file format")
        for _, row in dna_data.iterrows():
            position = row['position']
            if position in snp_list:
                genotype = row['genotype']
                print(f"Processing position {position} with genotype {genotype}")
                if genotype in ['AA', 'GG', 'TT', 'CC']:
                    genotype_dict[position] = 0
                elif genotype in ['AG', 'GA', 'CT', 'TC']:
                    genotype_dict[position] = 1
    
    if not genotype_dict:
        print("No valid genotype data found in the file")
        raise ValueError("No valid genotype data found in the file")

    feature_vector = pd.Series(genotype_dict, index=snp_list).fillna(7)
    matrix_for_prediction = pd.DataFrame([feature_vector])


    probabilities = model.predict_proba(matrix_for_prediction)[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_ethnicities = [(ethnicity_labels[i], probabilities[i]) for i in top_3_indices]

    return top_3_ethnicities
