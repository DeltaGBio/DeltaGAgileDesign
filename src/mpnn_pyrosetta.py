import os, re, glob, json, shutil, psutil
from itertools import tee

import tqdm, time
import pandas as pd, numpy as np
from scipy.spatial import cKDTree

from Bio import BiopythonWarning
from Bio.PDB import PDBParser, Selection, DSSP, Polypeptide, PDBIO, Select, Chain, Superimposer
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from colabdesign import clear_mem
from colabdesign.shared.utils import copy_dict    
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)
import gc

# Mapping from three-letter to one-letter amino acid codes
three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

class ProteinMPNN:
    """
    Class for using ProteinMPNN to redesign binders by fixing the residues at the interface.
    """
    
    def __init__(self, args: dict):
        """
        Initialize the ProteinMPNN class with the provided arguments.

        Args:
            args (dict): Dictionary containing necessary arguments and configurations.
        """
        self.args = args
    def create_output_dir(self):
        """
        Create the output directory for MPNN results.
        """
        os.makedirs(self.args["output_dir"], exist_ok=True)
    

    def parse_input_dir(self):
        """
        Parse the input directory to get the complex PDBs
        """
        complex_pdbs = glob.glob(f"{self.args['input_dir']}/*.pdb")
        return complex_pdbs

    
    def find_fix_residues(
        self,
        motifs: list[str],
        pdb_file: str,
        chain_id: str = "A"
    ) -> dict:
        """
        Find the positions of the residues to fix in the motifs.

        Inputs:
            motifs (list[str]): List of motif sequences to be fixed.
            pdb_file (str): Path to the PDB file containing the protein structure.
            chain_id (str): The chain identifier in the PDB file (default "A").

        Process:
            - Parses the PDB file and extracts the specified chain.
            - Constructs the chain's sequence from standard amino acids.
            - Searches for exact occurrences of each motif within the sequence.
            - Maps each motif occurrence to the corresponding residue IDs in the chain.

        Returns:
            dict: A dictionary mapping each motif to a list of occurrences, with each occurrence being a list of residue ID tuples.
        """
        
        # Parse the PDB structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_file)

        if chain_id not in structure[0]:
            logger.error(f"Chain {chain_id} not found in the PDB structure.")
            return {}
        chain = structure[0][chain_id]

        # Extract the chain sequence and record residue IDs
        chain_seq_list = []
        residue_id_list = []
        for residue in chain:
            if is_aa(residue, standard=True):
                try:
                    res_code = three_to_one_map[residue.get_resname()]
                except Exception:
                    res_code = "X"  # Unknown residue code
                chain_seq_list.append(res_code)
                residue_id_list.append(residue.get_id())
        chain_seq = "".join(chain_seq_list)

        motif_matches = {}
        # Search for each motif in the chain sequence and map residue positions to their one-letter codes
        residue_map = {}
        for motif in motifs:
            pattern = re.escape(motif)
            found = False
            for match in re.finditer(pattern, chain_seq):
                found = True
                start, end = match.start(), match.end()
                for idx in range(start, end):
                    res_id = residue_id_list[idx]
                    aa_code = chain_seq_list[idx]
                    # Modify key to use only the residue number
                    residue_map[res_id[1]] = aa_code
            if not found:
                logger.warning(f"Motif '{motif}' not found in chain {chain_id}.")
        return residue_map
    
    def condense_residue_ranges(self, residue_dict, chain_id):
        """
        Condense a dictionary of residue positions into a string of ranges.

        Args:
            residue_dict (dict): Dictionary with residue positions as keys.
            chain_id (str): Chain ID of the binder.

        Returns:
            str: Condensed string representation of residue ranges (e.g., "A55,A57-58,A61").
        """
        # Sort the residue positions
        positions = sorted(residue_dict.keys())

        # Return an empty string if no positions are available
        if not positions:
            return ""

        ranges = []
        start = positions[0]
        prev = start

        # Iterate through positions to form ranges
        for pos in positions[1:] + [None]:
            if pos != prev + 1:
                # Append the range or single position to the list
                if start == prev:
                    ranges.append(f"{chain_id}{start}")
                else:
                    ranges.append(f"{chain_id}{start}-{prev}")
                start = pos
            prev = pos

        # Join the ranges into a single string
        return ",".join(ranges)
    
    def find_residues_to_fix(
        self,
        motifs: list[str],
        pdb_file: str,
        chain_id: str = "A"
    ) -> dict[str, str]:
        """
        Identify residues within a specified distance from the binder chain in both motif A and B PDB files.

        Args:
            complex_pdb_motif_a (str): File path to the complex PDB for motif A.
            complex_pdb_motif_b (str): File path to the complex PDB for motif B.
            binder_chain (str): Identifier for the binder chain. Defaults to "B".
            atom_distance_cutoff (float): Maximum distance in angstroms to consider a residue as an interface. Defaults to 5.0.

        Returns:
            dict[str, str]: A dictionary of combined interface residues from both motifs, condensed into a string format.
        """
        fix_residues = self.find_fix_residues(
            motifs=motifs,
            pdb_file=pdb_file,
            chain_id=chain_id
        )
       
        # Condense the residues into a string
        fix_residues = self.condense_residue_ranges(
            fix_residues, 
            chain_id
        )
        print(fix_residues)
        return fix_residues
    
   
    def process_mpnn_sequences(self, mpnn_trajectories: dict) -> list[dict]:
        """
        Process MPNN sequences to filter out those with restricted amino acids.

        Args:
            mpnn_trajectories (dict): Dictionary containing MPNN sequence data.

        Returns:
            list[dict]: A list of dictionaries with filtered MPNN sequences.
        """
        # Create a set of restricted amino acids
        restricted_aas = {aa.strip().upper() for aa in self.args["omit_AAs"].split(",")}
        
        # Filter out sequences with restricted amino acids
        mpnn_sequences = sorted(
            (
                {
                    "seq": mpnn_trajectories["seq"][n],
                    "score": mpnn_trajectories["score"][n],
                    "seqid": mpnn_trajectories["seqid"][n],
                }
                for n in range(self.args["num_seqs"])
                if not restricted_aas or not any(
                    aa in mpnn_trajectories["seq"][n].split("/")[1].upper() for aa in restricted_aas
                )
            ),
            key=lambda x: x["score"]
        )

        # Filter out sequences that do not contain any of the present AAs
        present_aas = {aa.strip().upper() for aa in self.args["present_AAs"].split(",")}
        mpnn_sequences = [seq for seq in mpnn_sequences if any(aa in seq["seq"].split("/")[1].upper() for aa in present_aas)]
        
        return mpnn_sequences
    
    def mpnn_gen_sequence(self, complex_pdb: str, binder_chain: str = "B", interface_residues: str = "") -> list[str]:
        """
        Generate sequences for binders using MPNN.

        Args:
            complex_pdbs (list[str]): List of PDB file paths for complexes.
            binder_chain (str, optional): Chain identifier for the binder. Defaults to "B".
            interface_residues (str, optional): Residues at the interface to be fixed. Defaults to "".

        Returns:
            list[str]: List of unique MPNN-generated sequences.
        """
        from colabdesign.mpnn import mk_mpnn_model
        # Clear GPU memory to ensure efficient resource usage
        clear_mem()

        # Initialize MPNN model with specified parameters
        mpnn_model = mk_mpnn_model(
            backbone_noise=0,
            model_name=self.args["model_path"],
            weights=self.args["mpnn_weights"]
        )

        # Define design chains, including the binder chain
        design_chains = "A"

        # Determine fixed positions based on interface residues
        if interface_residues:
            fixed_positions = f"{interface_residues}"
            logger.info(f"Fixing interface residues: {fixed_positions}")
        else:
            logger.info("No interface residues to fix, skip this structure")
            return []
        print(fixed_positions)
        print(design_chains)
        print(complex_pdb)
        # Prepare inputs for MPNN model
        if self.args["omit_AAs"] != "":
            mpnn_model.prep_inputs(
                pdb_filename=complex_pdb,
                chain=design_chains,
                fix_pos=fixed_positions,
                rm_aa=self.args["omit_AAs"]
            )
        else:
            mpnn_model.prep_inputs(
                pdb_filename=complex_pdb,
                chain=design_chains,
                fix_pos=fixed_positions
            )

        # Sample MPNN sequences in parallel
        mpnn_sequences = mpnn_model.sample(
            temperature=self.args["sampling_temp_mpnn"],
            num=self.args["num_seqs"],
            batch=self.args["num_seqs"]
        )

        # Extract and deduplicate sequences
        list_mpnn_seq = list(mpnn_sequences["seq"])
        list_mpnn_seq = set(list_mpnn_seq)
        clear_mem()
        return list_mpnn_seq
    

    def generate_and_predict_sequences(self, complex_pdb: str):
        """
        Generate and predict sequences for a complex PDB
        """
        #get the name which 
        cycle_name = complex_pdb.split("/")[-1].split("_")[2].replace(".pdb", "")
        
        # Find the residues to fix and the pdb to build
        interface_residues_to_fix = self.find_residues_to_fix(
            motifs=self.args["motifs"],
            pdb_file=complex_pdb
        )

        # Generate sequences for binders using the structure with more hotspot residues
        mpnn_sequences = self.mpnn_gen_sequence(
            complex_pdb=complex_pdb,
            interface_residues=interface_residues_to_fix
        )
        binder_ids = [f"{cycle_name}_mpnn_{i}" for i in range(len(mpnn_sequences))]
        # Predict the structure of the designed proteins
        df_complex = self.predict_structure(
            binder_seqs=mpnn_sequences,
            binder_ids=binder_ids,
            origin_pdb=complex_pdb
        )
        
        return df_complex
        
    
    def predict_structure(self, binder_seqs: list[str], binder_ids: list[str], origin_pdb: str):
        """
        Predict the structure of the binder alone for the binder after redesigning by MPNN.
        """
        from colabdesign import mk_afdesign_model
        #clear the memory
        clear_mem()
        
        # Load AF2 model configurations
        models = self.args['models_binder']

        # Initialize the binder prediction model with specified parameters
        binder_prediction_model =  mk_afdesign_model(
            protocol="hallucination",
            use_templates=False,
            initial_guess=False,
            use_initial_atom_pos=False,
            num_recycles=self.args['num_recycles'],
            data_dir=self.args['af_params_dir'],
            use_multimer=False
        )
         

        #Initialize a list to store the binder seq and id
        list_binder_seqs = []
        list_binder_ids = []
        list_binder_ptms = []
        list_binder_plddt = []
        list_binder_pae = []
        list_binder_rmsd = []

        # Create the output directory for storing binder structure
        outdir_binder =  f"{self.args['output_dir']}" + "/structures"
        os.makedirs(outdir_binder, exist_ok=True)
        for binder_seq, binder_id in zip(binder_seqs, binder_ids):
            max_ptm = 0
            for model in models:
                binder_prediction_model.prep_inputs(length=len(binder_seq))
                binder_prediction_model.set_seq(binder_seq)
                binder_prediction_model.predict(models=[model], verbose=self.args['print_alphafold_stats'])
                prediction_metrics = copy_dict(binder_prediction_model.aux["log"])
                # Save the model with the highest ptm
                if prediction_metrics["ptm"] > self.args["ptm_threshold"] and prediction_metrics["ptm"] > max_ptm:
                    max_ptm = prediction_metrics["ptm"]
                    
                    pdb_path = os.path.join(outdir_binder, f"{binder_id}.pdb")
                    binder_prediction_model.save_pdb(
                        pdb_path
                    )

                    #calculate the RMSD before and after redesigning
                    rmsd_redesigned = self.RMSD_two_pdbs_using_motifs(
                        pdb_path,
                        origin_pdb,
                        self.args["motifs"]
                    )
                    
            #if there is at least one binder structure with ptm > ptm_threshold, add it to the list      
            if max_ptm > self.args["ptm_threshold"]:       
                list_binder_seqs.append(binder_seq)
                list_binder_ids.append(binder_id)
                list_binder_ptms.append(round(max_ptm, 3))
                list_binder_plddt.append(round(prediction_metrics["plddt"], 3))
                list_binder_pae.append(round(prediction_metrics["pae"], 3))
                list_binder_rmsd.append(round(rmsd_redesigned, 3))
        # Clear the memory to free up resources
        clear_mem()
        del binder_prediction_model
        gc.collect()
        
        #make a dataframe with the binder seq and id
        df_binder = pd.DataFrame({"id": list_binder_ids, "seq": list_binder_seqs, "binder_ptm": list_binder_ptms, "binder_plddt": list_binder_plddt, "binder_pae": list_binder_pae, "binder_rmsd": list_binder_rmsd})
        if df_binder.empty:
            logger.info(f"No designed binders passed the filter 'ptm_threshold' with value: {self.args['ptm_threshold']}")
            return None
        return df_binder
    
    # Helper function to extract Cα (CA) atoms from all standard amino acids in a chain.
    def get_ca_atoms(self, chain) -> list:
        """
        Returns a list of CA atoms for each residue in the chain that is a standard amino acid.
        This ensures that the RMSD calculation is based solely on the structural backbone.
        """
        return [residue["CA"] for residue in chain if "CA" in residue and is_aa(residue, standard=True)]
    
    def RMSD_two_pdbs_using_motifs(self, pdb_1: str, pdb_2: str, motifs: list) -> float:
        """
        Calculate the RMSD between two PDB files using motif-defined interface residues.

        This function:
        1. Identifies motif interface residue ranges in both PDBs.
        2. Extracts Cα atoms for those residues from chain A in each structure.
        3. Aligns the two sets of Cα atoms and computes the RMSD.

        Returns:
            float: The RMSD value between the motif regions of the two structures.
        """
        # Get motif residue ranges (e.g., "24-38,80-92") for each PDB, removing chain label
        motif_ranges_1 = self.find_residues_to_fix(motifs, pdb_1).replace("A", "")
        motif_ranges_2 = self.find_residues_to_fix(motifs, pdb_2).replace("A", "")

        # Parse structures and extract chain A
        parser = PDBParser(QUIET=True)
        chain_1 = parser.get_structure("structure_1", pdb_1)[0]["A"]
        chain_2 = parser.get_structure("structure_2", pdb_2)[0]["A"]

        # Get all Cα atoms for chain A in each structure
        ca_atoms_1 = self.get_ca_atoms(chain_1)
        ca_atoms_2 = self.get_ca_atoms(chain_2)

        # Helper to extract Cα atoms for all motif ranges
        def extract_atoms_by_ranges(ca_atoms, ranges_str):
            atoms = []
            for rng in ranges_str.split(","):
                if "-" in rng:
                    start, end = map(int, rng.split("-"))
                    # PDB residue numbering is 1-based, ca_atoms is 0-based
                    atoms.extend(ca_atoms[start-1:end])
                else:
                    idx = int(rng)
                    atoms.append(ca_atoms[idx-1])
            return atoms

        atoms_1_aligned = extract_atoms_by_ranges(ca_atoms_1, motif_ranges_1)
        atoms_2_aligned = extract_atoms_by_ranges(ca_atoms_2, motif_ranges_2)

        # Ensure both lists are the same length for alignment
        n = min(len(atoms_1_aligned), len(atoms_2_aligned))
        atoms_1_aligned = atoms_1_aligned[:n]
        atoms_2_aligned = atoms_2_aligned[:n]

        # Align and compute RMSD
        sup = Superimposer()
        sup.set_atoms(atoms_1_aligned, atoms_2_aligned)
        return round(sup.rms, 2)
