import glob
import argparse
import pandas as pd
import logging
from src.mpnn_pyrosetta import ProteinMPNN
import yaml
import os
def main(args):
    logging.basicConfig(filename='redesign.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    mpnn = ProteinMPNN(args)

    #create output directory
    mpnn.create_output_dir()

    #generate and predict sequences
    complex_pdbs = mpnn.parse_input_dir()
    #initialize a dataframe to store the results
    for complex_pdb in complex_pdbs:
        df = mpnn.generate_and_predict_sequences(complex_pdb)
        output_file = os.path.join(args["output_dir"], "redesign_results.csv")
        if os.path.exists(output_file):
            df.to_csv(output_file, index=False, mode='a', header=False)
        else:
            df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing the complex PDBs")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--motifs", nargs='+', type=str, required=True, help="Motifs to redesign, provided as a space-separated list of strings, e.g.: --motifs AECL AEFG")
    parser.add_argument("--setting_file", type=str, required=True, help="Path to the setting file")
    args = parser.parse_args()

    #load the setting file
    with open(args.setting_file, 'r') as f:
        setting = yaml.safe_load(f)

    #merge the setting file with the args
    args = {**setting, **vars(args)}

    main(args)