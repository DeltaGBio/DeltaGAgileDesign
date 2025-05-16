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
            binder_ids=binder_ids
        )
        
        return df_complex
        
    
    def predict_structure(self, binder_seqs: list[str], binder_ids: list[str]):
        """
        Predict the structure of the binder alone for the binder after redesigning by MPNN.
        """
        from colabdesign import mk_afdesign_model
        #clear the memory
        clear_mem()
        
        # Load AF2 model configurations
        models = self.args['models_binder']

        # Initialize the binder prediction model with specified parameters
        binder_prediction_model = mk_afdesign_model(
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
                    #calculate the RMSD of the binder structure alone and in the complex with SLIM A and SLIM B
                    binder_prediction_model.save_pdb(
                        os.path.join(outdir_binder, f"{binder_id}.pdb")
                    )
            #if there is at least one binder structure with ptm > ptm_threshold, add it to the list      
            if max_ptm > self.args["ptm_threshold"]:       
                list_binder_seqs.append(binder_seq)
                list_binder_ids.append(binder_id)
                list_binder_ptms.append(round(max_ptm, 3))
                list_binder_plddt.append(round(prediction_metrics["plddt"], 3))
                list_binder_pae.append(round(prediction_metrics["pae"], 3))
        # Clear the memory to free up resources
        clear_mem()
        del binder_prediction_model
        gc.collect()
        
        #make a dataframe with the binder seq and id
        df_binder = pd.DataFrame({"id": list_binder_ids, "seq": list_binder_seqs, "binder_ptm": list_binder_ptms, "binder_plddt": list_binder_plddt, "binder_pae": list_binder_pae})
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
    
    
    def calculate_binder_rmsd(self, df: pd.DataFrame, binder_chain: str = "A") -> dict:
        """
        df is a dataframe with the binder seq and id for binder alone and binder in the complex with SLIM A and SLIM B.
        
        Calculate the RMSD between the binder structure in isolation and its corresponding
        structures within complexes with SLIM A and SLIM B.

        This function extracts the specified binder chain from:
          - the standalone binder PDB (binder_alone),
          - the SLIM A complex PDB, and
          - the SLIM B complex PDB.
        
        It then computes the RMSD (based on Cα atoms) between the binder alone and each complex.
        Returns a dictionary with keys "SLIM_A" and "SLIM_B" for the respective RMSDs.
        """
        
        #get the binder pdb
        pdb_complex_dir = self.args["output_dir_mpnn_batch"] + "/complex_structure"
        
        binder_alone_dir = self.args["output_dir_mpnn_batch"] + "/binder_alone"
        
        #get the list of ids in the dataframe
        list_ids = df["id"].tolist()
        
        #initialize list to store the rmsd values
        rmsd_SLIM_As = []
        rmsd_SLIM_Bs = []
        ids = []
        for id in list_ids:
            binder_pdb = os.path.join(binder_alone_dir, f"{id}.pdb")
            complex_pdbs = [os.path.join(pdb_complex_dir, f"seq_a_{id}.pdb"), os.path.join(pdb_complex_dir, f"seq_b_{id}.pdb")]

            # Parse the binder structure and extract the specified chain.
            parser = PDBParser(QUIET=True)
            binder_structure = parser.get_structure("binder_alone", binder_pdb)
            #get the binder chain (always A chain)
            binder_alone_chain = binder_structure[0]["A"]
        
            # Parse the complex structures and extract the binder chain from each.
            complex_SLIM_A_structure = parser.get_structure("complex_SLIM_A", complex_pdbs[0])
            binder_SLIM_A_chain = complex_SLIM_A_structure[0][binder_chain]
        
            complex_SLIM_B_structure = parser.get_structure("complex_SLIM_B", complex_pdbs[1])
            binder_SLIM_B_chain = complex_SLIM_B_structure[0][binder_chain]
        
            # Extract CA atoms for the binder alone and binder in each complex.
            atoms_alone = self.get_ca_atoms(binder_alone_chain)
            atoms_SLIM_A = self.get_ca_atoms(binder_SLIM_A_chain)
            atoms_SLIM_B = self.get_ca_atoms(binder_SLIM_B_chain)
        
            
            #align the atoms
            def align_atoms(atoms1, atoms2):
                n = min(len(atoms1), len(atoms2))
                return atoms1[:n], atoms2[:n]
        
            atoms_alone_A, atoms_SLIM_A = align_atoms(atoms_alone, atoms_SLIM_A)
            atoms_alone_B, atoms_SLIM_B = align_atoms(atoms_alone, atoms_SLIM_B)
        
            # Calculate RMSD using the Bio.PDB Superimposer, which aligns two lists of atoms.
            sup = Superimposer()
            sup.set_atoms(atoms_alone_A, atoms_SLIM_A)
            rmsd_SLIM_A = round(sup.rms, 2)
        
            sup.set_atoms(atoms_alone_B, atoms_SLIM_B)
            rmsd_SLIM_B = round(sup.rms, 2)
        
            rmsd_SLIM_As.append(rmsd_SLIM_A)
            rmsd_SLIM_Bs.append(rmsd_SLIM_B)
            ids.append(id)
        
        #create a dataframe with the rmsd values and ids
        df_rmsd = pd.DataFrame({"id": ids, "SLIM_A_rmsd": rmsd_SLIM_As, "SLIM_B_rmsd": rmsd_SLIM_Bs})
        
        
        return df_rmsd
    
        
    def calculate_rmsd_complex_redesign(self, df: pd.DataFrame, ) -> dict:
        """
        Calculate the RMSD between the complex before and after redesigning the binder by proteinMPNN.
        """
        #get the list of ids
        validation_dir = self.args['output_dir_validation_batch'] + "/complex_structure"
        pdb_complex_dir = self.args["output_dir_mpnn_batch"] + "/complex_structure"
        
        #convert the df to a list of ids
        list_ids = df["id"].tolist()   
        
           #initialize list to store the rmsd values
        rmsds_SLIM_A_before_after = []
        rmsds_SLIM_B_before_after = []
        ids = []
        
        
        for id in list_ids:
            
            #get the pdb files after redesigning
            SLIM_A_after_redesign_pdb = os.path.join(pdb_complex_dir, f"seq_a_{id}.pdb")
            SLIM_B_after_redesign_pdb = os.path.join(pdb_complex_dir, f"seq_b_{id}.pdb")
            
            #get the pdb files before redesigning
            id_before_redesign = id.split("_mpnn")[0]
            
            # Query and get the pdb files before the design
            SLIM_A_before_redesign_pdb = glob.glob(os.path.join(validation_dir, f"seq_a_{id_before_redesign}_*.pdb"))[0]
            SLIM_B_before_redesign_pdb = glob.glob(os.path.join(validation_dir, f"seq_b_{id_before_redesign}_*.pdb"))[0]
            
            #parse the pdb files
            parser = PDBParser(QUIET=True)

            #before
            SLIM_A_before_redesign_structure = parser.get_structure("SLIM_A_before_redesign", SLIM_A_before_redesign_pdb)
            SLIM_B_before_redesign_structure = parser.get_structure("SLIM_B_before_redesign", SLIM_B_before_redesign_pdb)
            
            
            #after
            SLIM_A_after_redesign_structure = parser.get_structure("SLIM_A_after_redesign", SLIM_A_after_redesign_pdb)
            SLIM_B_after_redesign_structure = parser.get_structure("SLIM_B_after_redesign", SLIM_B_after_redesign_pdb)
            
            #get the CA atoms of both chains
            atoms_SLIM_A_before = self.get_ca_atoms(SLIM_A_before_redesign_structure[0]['A']) + self.get_ca_atoms(SLIM_A_before_redesign_structure[0]['B'])
            atoms_SLIM_B_before = self.get_ca_atoms(SLIM_B_before_redesign_structure[0]['A']) + self.get_ca_atoms(SLIM_B_before_redesign_structure[0]['B'])
            atoms_SLIM_A_after = self.get_ca_atoms(SLIM_A_after_redesign_structure[0]['A']) + self.get_ca_atoms(SLIM_A_after_redesign_structure[0]['B'])
            atoms_SLIM_B_after = self.get_ca_atoms(SLIM_B_after_redesign_structure[0]['A']) + self.get_ca_atoms(SLIM_B_after_redesign_structure[0]['B'])
            
            # Calculate RMSD using the Bio.PDB Superimposer, which aligns two lists of atoms.
            sup = Superimposer()
            sup.set_atoms(atoms_SLIM_A_before, atoms_SLIM_A_after)
            rmsd_SLIM_A_before_after = round(sup.rms, 2)
        
            sup.set_atoms(atoms_SLIM_B_before, atoms_SLIM_B_after)
            rmsd_SLIM_B_before_after = round(sup.rms, 2)
        
            rmsds_SLIM_A_before_after.append(rmsd_SLIM_A_before_after)
            rmsds_SLIM_B_before_after.append(rmsd_SLIM_B_before_after)
            ids.append(id)
            
        #create a dataframe with the rmsd values and ids
        df_rmsd_before_after = pd.DataFrame({"id": ids, "SLIM_A_rmsd_before_after": rmsds_SLIM_A_before_after, "SLIM_B_rmsd_before_after": rmsds_SLIM_B_before_after})
        
        return df_rmsd_before_after
            
    
    def process_and_save_binders(self, merged_df: pd.DataFrame) -> None:
        """
        Process the merged DataFrame to filter binders based on RMSD thresholds,
        remove unnecessary files, and save the results to a CSV file.

        Args:
            merged_df (pd.DataFrame): DataFrame containing binder and complex information.
            mpnn (ProteinMPNN): Instance of ProteinMPNN containing necessary arguments.
        """
        # Filter out the binders that do not meet the RMSD threshold
        rmsd_threshold = self.args["RMSD_binder"]
        merged_df = merged_df[(merged_df["SLIM_A_rmsd"] < rmsd_threshold) & 
                                (merged_df["SLIM_B_rmsd"] < rmsd_threshold)]
        # If the df_merged is empty, return None
        if merged_df.empty:
            logger.info(f"No designed binders passed the filter 'RMSD_binder' with value: {rmsd_threshold}")
            return None
        
        # Filter out the binders that do not meet the RMSD threshold after redesigning
        rmsd_threshold_redesign = self.args["RMSD_complex_redesign"]
        merged_df = merged_df[(merged_df["SLIM_A_rmsd_before_after"] < rmsd_threshold_redesign) & 
                                (merged_df["SLIM_B_rmsd_before_after"] < rmsd_threshold_redesign)]
        # If the df_merged is empty, return None
        if merged_df.empty:
            logger.info(f"No designed binders passed the filter 'RMSD_complex_redesign' with value: {rmsd_threshold_redesign}")
            return None
        
        keep_ids = merged_df["id"].tolist()
        # Function to remove files not in keep_ids
        def remove_unwanted_files(directory: str, keep_ids: list) -> None:
            for file in os.listdir(directory):
                file_id = file.split("/")[-1].split(".")[0]
                file_id = file_id.replace("seq_a_", "").replace("seq_b_", "")
                if file_id not in keep_ids:
                    os.remove(os.path.join(directory, file))
        
        # Remove unwanted files in complex_structure and binder_alone directories
        remove_unwanted_files(self.args["output_dir_mpnn_batch"] + "/complex_structure", keep_ids)
        remove_unwanted_files(self.args["output_dir_mpnn_batch"] + "/binder_alone", keep_ids)
        
        # Save the merged DataFrame
        report_csv = os.path.join(self.args["output_dir_mpnn"], "mpnn_redesigned_binders.csv")
        if not os.path.exists(report_csv):
            merged_df.to_csv(report_csv, index=False)
        else:
            merged_df.to_csv(report_csv, mode='a', header=False, index=False)
        return merged_df
            

import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
            
class PyrosettaValidation:
    
    def __init__(self, args: dict) -> None:
        self.args = args
    
    def init_pyrosetta(self, batch_idx: int):
        """
        Initialize PyRosetta for the first batch
        """
        #if the pyrosetta is not initialized, initialize it
        if batch_idx == 0 or not self.args.get("pyrosetta_initialized", False):
            pr.init(options=(
                "-mute all "
                "-ignore_unrecognized_res -ignore_zero_occupancy "
                f"-holes:dalphaball {self.args['dalphaball_path']} "
                "-corrections::beta_nov16 true -relax:default_repeats 1 "
                        ))
            self.args["pyrosetta_initialized"] = True
        
    def create_output_dir(self):
        output_dir = os.path.join(self.args["out_dir"], "pyrosetta")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.args["output_dir_pyrosetta"] = output_dir
        
        output_dir_batch = os.path.join(output_dir, self.args["batch_name"])
        if not os.path.exists(output_dir_batch):
            os.makedirs(output_dir_batch)
        self.args["output_dir_pyrosetta_batch"] = output_dir_batch
        
    def relax_pdb_by_pyrosetta(self, pdb_file, relaxed_pdb_path):
        """
        Relax a PDB file using PyRosetta.
        """
        try:
            #get the pose
            pose = pr.pose_from_pdb(pdb_file)
        except Exception as e:
            logger.error(f"Error loading PDB file: {e}")
            return None
        
        start_pose = pose.clone()

        # Generate movemaps
        mmf = MoveMap()
        mmf.set_chi(True)  # Enable sidechain movement
        mmf.set_bb(True)  # Enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
        mmf.set_jump(False)  # Disable whole chain movement

        # Run FastRelax
        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf)  # Set MoveMap
        fastrelax.max_iter(200)  # Default iterations is 2500
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)

        # Align relaxed structure to original trajectory
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)

        # Copy B factors from start_pose to pose
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                # Get the B factor of the first heavy atom in the residue
                bfactor = start_pose.pdb_info().bfactor(resid, 1)
                for atom_id in range(1, pose.residue(resid).natoms() + 1):
                    pose.pdb_info().bfactor(resid, atom_id, bfactor)

        # Output relaxed and aligned PDB
        #get the file_name of the pdb file
        file_name = os.path.basename(pdb_file)
        relaxed_pdb_path = os.path.join(relaxed_pdb_path, "relaxed_"+file_name)
        
        pose.dump_pdb(relaxed_pdb_path)
              
        return relaxed_pdb_path

    def pdb_to_string(self, pdb_file, chains=None, models=None):
        """Read pdb file and return as string."""
        if chains is not None:
            if "," in chains:
                chains = chains.split(",")
            if not isinstance(chains, list):
                chains = [chains]
        if models is None:
            models = [1]
        elif not isinstance(models, list):
            models = [models]

        #get the MODRES dictionary
        MODRES = {
            'MSE': 'MET', 'MLY': 'LYS', 'FME': 'MET', 'HYP': 'PRO',
            'TPO': 'THR', 'CSO': 'CYS', 'SEP': 'SER', 'M3L': 'LYS',
            'HSK': 'HIS', 'SAC': 'SER', 'PCA': 'GLU', 'DAL': 'ALA',
            'CME': 'CYS', 'CSD': 'CYS', 'OCS': 'CYS', 'DPR': 'PRO',
            'B3K': 'LYS', 'ALY': 'LYS', 'YCM': 'CYS', 'MLZ': 'LYS',
            '4BF': 'TYR', 'KCX': 'LYS', 'B3E': 'GLU', 'B3D': 'ASP',
            'HZP': 'PRO', 'CSX': 'CYS', 'BAL': 'ALA', 'HIC': 'HIS',
            'DBZ': 'ALA', 'DCY': 'CYS', 'DVA': 'VAL', 'NLE': 'LEU',
            'SMC': 'CYS', 'AGM': 'ARG', 'B3A': 'ALA', 'DAS': 'ASP',
            'DLY': 'LYS', 'DSN': 'SER', 'DTH': 'THR', 'GL3': 'GLY',
            'HY3': 'PRO', 'LLP': 'LYS', 'MGN': 'GLN', 'MHS': 'HIS',
            'TRQ': 'TRP', 'B3Y': 'TYR', 'PHI': 'PHE', 'PTR': 'TYR',
            'TYS': 'TYR', 'IAS': 'ASP', 'GPL': 'LYS', 'KYN': 'TRP',
            'CSD': 'CYS', 'SEC': 'CYS'
        } 
        modres = {**MODRES}
        lines = []
        seen = []
        model = 1
        with open(pdb_file, "rb") as file:
            for line in file:
                line = line.decode("utf-8", "ignore").rstrip()
                if line.startswith("MODEL"):
                    model = int(line[5:])
                if models is None or model in models:
                    if line.startswith("MODRES"):
                        k = line[12:15]
                        v = line[24:27]
                        if k not in modres and v in residue_constants.restype_3to1:
                            modres[k] = v
                    if line.startswith("HETATM"):
                        k = line[17:20]
                        if k in modres:
                            line = "ATOM  " + line[6:17] + modres[k] + line[20:]
                    if line.startswith("ATOM"):
                        chain = line[21:22]
                        if chains is None or chain in chains:
                            atom = line[12:16].strip()
                            resi = line[17:20]
                            resn = line[22:27].strip()
                            if resn[-1].isalpha():  # alternative atom
                                resn = resn[:-1]
                                line = line[:26] + " " + line[27:]
                            key = f"{model}_{chain}_{resn}_{resi}_{atom}"
                            if key not in seen:  # skip alternative placements
                                lines.append(line)
                                seen.append(key)
                    if line.startswith("MODEL") or line.startswith("TER") or line.startswith("ENDMDL"):
                        lines.append(line)
        return "\n".join(lines)


    def relax_pdb_by_amber(self, pdb_file, relaxed_pdb_path, max_iterations=10000, tolerance =2.39, stiffness=10.0, use_gpu=True):
        """Relax the protein structure using AmberRelaxation."""
        pdb_str = self.pdb_to_string(pdb_file)
        protein_obj = protein.from_pdb_string(pdb_str)
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=max_iterations,
            tolerance=tolerance,
            stiffness=stiffness,
            exclude_residues=[],
            max_outer_iterations=3,
            use_gpu=use_gpu
        )
        relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=protein_obj)
        
        #get the file_name of the pdb file
        file_name = os.path.basename(pdb_file)
        pdb_out = os.path.join(relaxed_pdb_path, "relaxed_"+file_name)
        
        #dump the relaxed pdb
        with open(pdb_out, 'w') as f:
            f.write(relaxed_pdb_lines)
            
        return pdb_out 

    
    def find_interface_residues(
        self, 
        complex_pdb: str, 
        binder_chain: str = "B", 
        target_chain: str = "A",
        atom_distance_cutoff: float = 5.0
    ) -> dict:
        """
        Identify residues within a specified distance from the binder chain in a PDB structure.

        This function parses a PDB file to find residues in the target chain that are within a 
        given distance from the specified binder chain. It returns a dictionary mapping residue 
        positions to their one-letter amino acid codes.

        Args:
            complex_pdb (str): Path to the complex PDB file.
            binder_chain (str, optional): Chain ID of the binder. Defaults to "B".
            atom_distance_cutoff (float, optional): Distance cutoff in angstroms. Defaults to 5.0.

        Returns:
            dict: A dictionary mapping residue positions to their one-letter amino acid codes.
        """
        
        # Mapping from three-letter to one-letter amino acid codes
        three_to_one_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        # Parse the PDB file to extract the structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", complex_pdb)

        # Extract atoms from the specified binder chain
        binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
        binder_coords = np.array([atom.coord for atom in binder_atoms])

        # Extract atoms from the target chain (assumed to be 'A')
        target_atoms = Selection.unfold_entities(structure[0][target_chain], 'A')
        target_coords = np.array([atom.coord for atom in target_atoms])

        # Build KD trees for efficient spatial queries
        binder_tree = cKDTree(binder_coords)
        target_tree = cKDTree(target_coords)

        # Dictionary to store interacting residues
        interacting_residues = {}

        # Find pairs of atoms within the specified distance cutoff
        pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

        # Process each binder atom's interactions
        for binder_idx, close_indices in enumerate(pairs):
            binder_residue = binder_atoms[binder_idx].get_parent()
            binder_resname = binder_residue.get_resname()

            # Convert three-letter code to single-letter code
            if binder_resname in three_to_one_map:
                aa_single_letter = three_to_one_map[binder_resname]
                for close_idx in close_indices:
                    target_residue = target_atoms[close_idx].get_parent()
                    interacting_residues[binder_residue.id[1]] = aa_single_letter

        return interacting_residues

    # Rosetta interface scores
    def score_interface(self, pdb_file, binder_chain="B", target_chain="A", prefix=""):
        """
        Calculate Rosetta interface scores for a given PDB file.

        Args:
            pdb_file (str): Path to the PDB file.
            binder_chain (str, optional): Chain ID of the binder. Defaults to "B".
            target_chain (str, optional): Chain ID of the target. Defaults to "A".

        Returns:
            tuple: A tuple containing interface scores, amino acid counts at the interface,
                   and a comma-separated string of interface residue PDB IDs.
        """
        # Load the pose and configure the interface analyzer
        pose = pr.pose_from_pdb(pdb_file)
        iam = InterfaceAnalyzerMover()
        iam.set_interface("A_B")
        scorefxn = pr.get_fa_scorefxn()
        iam.set_scorefunction(scorefxn)
        iam.set_compute_packstat(True)
        iam.set_compute_interface_energy(True)
        iam.set_calc_dSASA(True)
        iam.set_calc_hbond_sasaE(True)
        iam.set_compute_interface_sc(True)
        iam.set_pack_separated(True)
        iam.apply(pose)

        # Identify interface residues and count amino acid occurrences
        interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
        residue_map = self.find_interface_residues(pdb_file, binder_chain=binder_chain, target_chain=target_chain)
        pdb_ids = []
        for res_num, aa in residue_map.items():
            interface_AA[aa] += 1
            pdb_ids.append(f"{binder_chain}{res_num}")
        n_res = len(pdb_ids)
        pdb_ids_str = ",".join(pdb_ids)
        
        # Compute interface hydrophobicity percentage
        hydrophobic_count = sum(interface_AA[aa] for aa in set("ACFILMPVWY"))
        interface_hydrophobicity = (hydrophobic_count / n_res * 100) if n_res else 0

        # Retrieve various interface statistics
        data = iam.get_all_data()
        interface_sc = data.sc_value
        interface_hbonds = data.interface_hbonds
        interface_dG = iam.get_interface_dG()
        interface_dSASA = iam.get_interface_delta_sasa()
        interface_packstat = iam.get_interface_packstat()
        interface_dG_SASA_ratio = data.dG_dSASA_ratio * 100
        buns_filter = XmlObjects.static_get_filter(
            '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" '
            'ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" '
            'probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
        )
        interface_delta_unsat_hbonds = buns_filter.report_sm(pose)
        interface_hbond_percentage = (interface_hbonds / n_res * 100) if n_res else None
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / n_res * 100) if n_res else None

        # Calculate binder energy and SASA-based metrics
        chain_sel = ChainSelector(binder_chain)
        tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
        tem.set_scorefunction(scorefxn)
        tem.set_residue_selector(chain_sel)
        binder_score = tem.calculate(pose)

        bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
        bsasa.set_residue_selector(chain_sel)
        binder_sasa = bsasa.calculate(pose)
        interface_binder_fraction = (interface_dSASA / binder_sasa * 100) if binder_sasa > 0 else 0

        # Determine the binder pose by finding the matching chain
        binder_pose = None
        for i, chain in enumerate(pose.split_by_chain(), start=1):
            if pose.pdb_info().chain(pose.conformation().chain_begin(i)) == binder_chain:
                binder_pose = chain
                break

        # Compute surface hydrophobicity of the binder chain
        layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)
        exp_apol = total = 0
        for i in range(1, len(surface_res) + 1):
            if surface_res[i]:
                res = binder_pose.residue(i)
                if res.is_apolar() or res.name() in {"PHE", "TRP", "TYR"}:
                    exp_apol += 1
                total += 1
        surface_hydrophobicity = exp_apol / total if total else 0

        # Compile and round all interface scores
        scores = {
            f"{prefix}_binder_score": binder_score,
            f"{prefix}_surface_hydrophobicity": surface_hydrophobicity,
            f"{prefix}_interface_sc": interface_sc,
            f"{prefix}_interface_packstat": interface_packstat,
            f"{prefix}_interface_dG": interface_dG,
            f"{prefix}_interface_dSASA": interface_dSASA,
            f"{prefix}_interface_dG_SASA_ratio": interface_dG_SASA_ratio,
            f"{prefix}_interface_fraction": interface_binder_fraction,
            f"{prefix}_interface_hydrophobicity": interface_hydrophobicity,
            f"{prefix}_interface_nres": n_res,
            f"{prefix}_interface_interface_hbonds": interface_hbonds,
            f"{prefix}_interface_hbond_percentage": interface_hbond_percentage,
            f"{prefix}_interface_delta_unsat_hbonds": interface_delta_unsat_hbonds,
            f"{prefix}_interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,
            f"{prefix}_interface_residues": pdb_ids_str,
            f"{prefix}_interface_AA": interface_AA
        }
        scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in scores.items()}

        return scores
        
    def calc_ss_percentage(self, pdb_file, chain_id="B", atom_distance_cutoff=5.0, prefix=""):
        """Calculate secondary structure percentages and average pLDDT scores for a given PDB file.
        
        This function:
          - Parses the PDB file to extract the protein structure (using only the first model).
          - Uses DSSP to assign secondary structures to residues.
          - Identifies interface residues via self.find_interface_residues.
          - Computes the percentages of helix, sheet, and loop for the whole chain and the interface.
          - Averages the normalized pLDDT scores (from bfactor values) for structured and interface residues.
        
        
        """
        # Parse the protein structure (first model only)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        
        # Run DSSP to assign secondary structure to each residue
        dssp = DSSP(model, pdb_file, dssp=self.args["dssp_path"])
        
        # Initialize counters and lists for residue classification and pLDDT scores
        ss_counts = defaultdict(int)            # Overall counts of helix, sheet, and loop
        interface_ss_counts = defaultdict(int)    # Counts restricted to interface residues
        plddts_ss = []         # Normalized pLDDT scores for non-loop residues
        plddts_interface = []  # Normalized pLDDT scores for interface residues
        
        # Access the specified chain and determine its interface residues
        chain = model[chain_id]
        interacting_residues = set(self.find_interface_residues(pdb_file))
        
        # Process each residue in the chain
        for residue in chain:
            res_num = residue.id[1]  # Extract residue sequence number
            # Proceed only if the residue has a DSSP assignment
            if (chain_id, res_num) in dssp:
                ss_code = dssp[(chain_id, res_num)][2]  # Secondary structure code from DSSP
                # Map DSSP code to simplified structure type (default is 'loop')
                if ss_code in ['H', 'G', 'I']:
                    ss_type = 'helix'
                elif ss_code == 'E':
                    ss_type = 'sheet'
                else:
                    ss_type = 'loop'
                
                ss_counts[ss_type] += 1  # Count overall secondary structure occurrence
                
                # For structured residues (helix or sheet), compute and store the average pLDDT score
                if ss_type != 'loop':
                    residue_plddt = sum(atom.bfactor for atom in residue) / len(residue)
                    plddts_ss.append(residue_plddt)
                
                # If the residue is part of the interface, update counts and pLDDT for the interface
                if res_num in interacting_residues:
                    interface_ss_counts[ss_type] += 1
                    residue_plddt = sum(atom.bfactor for atom in residue) / len(residue)
                    plddts_interface.append(residue_plddt)
        
        # Calculate total counts of residues overall and at the interface
        total_residues = sum(ss_counts.values())
        total_interface = sum(interface_ss_counts.values())
        
        # Compute secondary structure percentages for the overall chain
        if total_residues:
            helix_pct = round((ss_counts['helix'] / total_residues) * 100, 2)
            sheet_pct = round((ss_counts['sheet'] / total_residues) * 100, 2)
            loop_pct = round(((total_residues - ss_counts['helix'] - ss_counts['sheet']) / total_residues) * 100, 2)
            percentages = (helix_pct, sheet_pct, loop_pct)
        else:
            percentages = (0, 0, 0)
        
        # Compute secondary structure percentages for interface residues
        if total_interface:
            int_helix_pct = round((interface_ss_counts['helix'] / total_interface) * 100, 2)
            int_sheet_pct = round((interface_ss_counts['sheet'] / total_interface) * 100, 2)
            int_loop_pct = round(((total_interface - interface_ss_counts['helix'] - interface_ss_counts['sheet']) / total_interface) * 100, 2)
            interface_percentages = (int_helix_pct, int_sheet_pct, int_loop_pct)
        else:
            interface_percentages = (0, 0, 0)
        
        # Compute average normalized pLDDT scores (dividing by 100 to scale appropriately)
        avg_i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
        avg_ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0
        
        # Consolidate results into a dictionary for downstream analysis
        ss_results = {
            f"{prefix}_i_plddt": avg_i_plddt,                         # Average normalized pLDDT for interface residues
            f"{prefix}_ss_plddt": avg_ss_plddt,                        # Average normalized pLDDT for helix and sheet regions
            f"{prefix}_binder_percentages_helix": percentages[0],                     # (helix, sheet, loop) percentages overall
            f"{prefix}_binder_percentages_sheet": percentages[1],                     # (helix, sheet, loop) percentages overall
            f"{prefix}_binder_percentages_loop": percentages[2],                     # (helix, sheet, loop) percentages overall
            f"{prefix}_interface_percentages_helix": interface_percentages[0], # (helix, sheet, loop) percentages at the interface
            f"{prefix}_interface_percentages_sheet": interface_percentages[1], # (helix, sheet, loop) percentages at the interface
            f"{prefix}_interface_percentages_loop": interface_percentages[2], # (helix, sheet, loop) percentages at the interface
        }
        
        return ss_results
        
    def process_single_design(self, row, relaxed_path):
        """Process a single design through PyRosetta validation"""
        results = {}
        
        # Process both SLIM A and B structures
        for slim in ['a', 'b']:
            # Prepare and relax PDB
            pdb_file = os.path.join(self.args["output_dir_mpnn_batch"], 
                                    "complex_structure", 
                                    f"seq_{slim}_{row['id']}.pdb")
            if self.args["relax_pdb"] == "pyrosetta":
                relaxed_pdb = self.relax_pdb_by_pyrosetta(pdb_file, relaxed_path)
            if self.args["relax_pdb"] == "amber":
                relaxed_pdb = self.relax_pdb_by_amber(pdb_file, relaxed_path)

            # Skip the function if the PDB file is not found
            if relaxed_pdb is None:
                return None
            
            # Calculate scores
            prefix = f"SLIM_{slim.upper()}"
            results.update(self.get_plddt_score_from_pdb(relaxed_pdb, prefix=prefix))
            results.update(self.calc_ss_percentage(relaxed_pdb, prefix=prefix))
            results.update(self.score_interface(relaxed_pdb, prefix=prefix))
            results.update(self.calculate_clash_score(relaxed_pdb, prefix=prefix))
        return results

    def check_filters(self, results_df, filter_rules):
        """Check if design passes all filters"""
        for filter_dict in filter_rules:
            filter_name = list(filter_dict.keys())[0]
            filter_range = list(filter_dict.values())[0]
            
            for prefix in ["SLIM_A", "SLIM_B"]:
                value = float(results_df.loc[0, f"{prefix}_{filter_name}"])
                if not (filter_range[0] <= value <= filter_range[1]):
                    logger.warning(f"Design {prefix}_{results_df.loc[0, 'id']} failed the filter {filter_name} with value: {value}")
                    return False
        return True
    
    def calculate_clash_score(self, pdb_file, threshold=2.4, only_ca=False, prefix=""):
        """Calculate the clash score for a given PDB file."""
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        atoms = []
        atom_info = []  # Detailed atom info for debugging and processing

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element == 'H':  # Skip hydrogen atoms
                            continue
                        if only_ca and atom.get_name() != 'CA':
                            continue
                        atoms.append(atom.coord)
                        atom_info.append(
                            (chain.id, residue.id[1], atom.get_name(), atom.coord)
                        )

        tree = cKDTree(atoms)
        pairs = tree.query_pairs(threshold)

        valid_pairs = set()
        for i, j in pairs:
            chain_i, res_i, name_i, coord_i = atom_info[i]
            chain_j, res_j, name_j, coord_j = atom_info[j]

            # Exclude clashes within the same residue
            if chain_i == chain_j and res_i == res_j:
                continue

            # Exclude directly sequential residues in the same chain for all atoms
            if chain_i == chain_j and abs(res_i - res_j) == 1:
                continue

            # If calculating sidechain clashes, only consider clashes between different chains
            if not only_ca and chain_i == chain_j:
                continue

            valid_pairs.add((i, j))

        return {f"{prefix}_num_clashes": len(valid_pairs)}
    
    def get_plddt_score_from_pdb(self, pdb_file: str, prefix, chain_id: str = "A") -> dict:
        """Get the pLDDT score from CA atoms in a PDB file for a specified chain."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        chain = model[chain_id]
        
        plddt_scores = [
            atom.bfactor for residue in chain for atom in residue if atom.get_name() == "CA"
        ]  # Collect pLDDT scores from CA atoms
        
        if not plddt_scores:
            raise ValueError(f"No pLDDT scores found in chain {chain_id} of the PDB file.")
        
        average_plddt = round(sum(plddt_scores) / len(plddt_scores), 2)  # Calculate the average pLDDT score
        return {f"{prefix}_plddt": average_plddt}
        
