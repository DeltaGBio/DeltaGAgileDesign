#!/bin/bash
#SBATCH --job-name=redesign
#SBATCH --partition=cosb
#SBATCH --output=logs/redesign_%j.out
#SBATCH --error=logs/redesign_%j.err
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8   
#SBATCH --gres=gpu:1

module load apptainer
sif="/pasteur/appa/scratch/dvu/github/SLIMshot/apptainer/SLIMshot_20250429.sif"
input="RFdiffusion/20250521_EGFR/pdb_outputs_domain"
output="AgileDesign/EGFR"
motifs="KVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTV"
setting_file="setting/setting.yaml"
apptainer exec --nv $sif bash -c "python redesign.py --input_dir $input --output_dir $output --motifs $motifs --setting_file $setting_file"