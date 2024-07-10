import pandas as pd
from pyliftover import LiftOver
from loguru import logger


# Setup logging

def convert_to_hg38(dna_file_path, output_file_path, from_assembly='hg19'):
    # Load the appropriate chain file for LiftOver
    lo = LiftOver(from_assembly, 'hg38')

    # Read the first few lines of the file to detect format
    with open(dna_file_path, 'r') as file:
        first_lines = [file.readline().strip() for _ in range(10)]

    # Detect format based on the first line
    if any("AncestryDNA" in line for line in first_lines):
        logger.info("Detected AncestryDNA format")
        dna_data = pd.read_table(dna_file_path, sep='\t', comment='#', names=["rsid", "chromosome", "position", "allele1", "allele2"], skiprows=1)
        dna_data['genotype'] = dna_data['allele1'] + dna_data['allele2']
        dna_data.drop(columns=['allele1', 'allele2'], inplace=True)
        format_type = "AncestryDNA"
    elif any("23andMe" in line for line in first_lines):
        logger.info("Detected 23andMe format")
        dna_data = pd.read_table(dna_file_path, sep='\t', comment='#', names=["rsid", "chromosome", "position", "genotype"], skiprows=1)
        format_type = "23andMe"
    else:
        # General format
        logger.info("Detected general DNA format")
        dna_data = pd.read_table(dna_file_path, sep=',', comment='#', names=["rsid", "chromosome", "position", "genotype"], skiprows=1)
        # Print the columns for verification
        print("Columns: ", dna_data.head(10))
        if not 'genotype' in dna_data.columns:
            raise ValueError("Expected columns 'rsid', 'chromosome', 'position', 'genotype' not found in the file")
        format_type = "General"
    
    if not all(col in dna_data.columns for col in ['rsid', 'chromosome', 'position', 'genotype']):
        raise ValueError("Expected columns 'rsid', 'chromosome', 'position', 'genotype' not found in the file")

    # Remove rows with missing or malformed data in 'position' column
    dna_data.dropna(subset=['position'], inplace=True)
    dna_data = dna_data[dna_data['position'].apply(lambda x: str(x).isdigit())]

    # Perform LiftOver for each position
    new_positions = []
    for _, row in dna_data.iterrows():
        chrom = str(row['chromosome']).strip()
        pos = int(row['position'])
        new_pos = lo.convert_coordinate(chrom, pos)
        if new_pos:
            # Take the first mapping if available
            new_positions.append(int(new_pos[0][1]))
        else:
            new_positions.append(pos)  # Keep the original position if LiftOver fails

    dna_data['position'] = new_positions

    # Prepare header comment based on the format type
    header_comment = f"# {format_type} format\n"

    # Save the new DataFrame to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.write(header_comment)
        dna_data.to_csv(output_file, sep='\t', index=False)
    
    logger.info(f"Converted DNA file saved to {output_file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a DNA file to hg38 using pyliftover")
    parser.add_argument("input_file", help="Path to the input DNA file")
    parser.add_argument("output_file", help="Path to the output DNA file")
    parser.add_argument("from_assembly", help="Original assembly version (e.g., hg19, hg18)")

    args = parser.parse_args()

    convert_to_hg38(args.input_file, args.output_file, args.from_assembly)
