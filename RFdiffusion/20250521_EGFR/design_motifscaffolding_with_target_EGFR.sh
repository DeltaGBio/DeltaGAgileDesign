#!/bin/bash
# Here, we're running one of the motif-scaffolding benchmark examples, in the presence of a target
# Specifically, we're scaffolding the Mdm2-interacting peptide from p53
# We specify the output path and input pdb (the p53-Mdm2 complex)
# We specify the protein we want to build, with the contig input:
#   - the Mdm2 target protein (residues A25-109), with a chain break.
#   - 0-70 residues (randomly sampled)
#   - residues 17-29 (inclusive) on the B chain of the input (the p53 helix)
#   - 0-70 residues (randomly sampled)
# We also constrain the total length of the diffused chain to be within 70 and 120 residues
# We generate 10 designs
# As in the paper (at least for some of the designs we tested), we use the complex-finetuned model
#!/usr/bin/bash
#SBATCH --output=logs/%j_design_partialdiffusion.out  # Append job ID to output file for uniqueness
#SBATCH --error=logs/%j_design_partialdiffusion.err
#SBATCH --partition=cosb
#SBATCH --gres=gpu:1        # remember 1 GPU
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

module load apptainer

sif="/pasteur/appa/scratch/dvu/github/RFdiffusion/apptainer/RFdiffusion_20250509.sif"
src_dir="/pasteur/appa/scratch/dvu/github/RFdiffusion/run_inference.py"
apptainer exec --nv -B /pasteur  "${sif}" bash -c \
    "python3.9 ${src_dir} \
    inference.output_prefix=pdb_outputs_domain/EGFR_domain \
    inference.input_pdb=input_pdbs/4uv7.pdb \
    'contigmap.contigs=[50-100/A311-374/50-100]' \
    contigmap.length=100-200  inference.num_designs=1000 inference.ckpt_override_path=../../models/Complex_base_ckpt.pt"

