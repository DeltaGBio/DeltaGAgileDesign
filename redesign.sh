#!/bin/bash
#SBATCH --job-name=redesign
#SBATCH --partition=cyaa
#SBATCH --output=logs/redesign_%j.out
#SBATCH --error=logs/redesign_%j.err
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8   
#SBATCH --gres=gpu:1

module load apptainer
sif="/pasteur/appa/scratch/dvu/github/SLIMshot/apptainer/SLIMshot_20250429.sif"
input="RFdiffusion/Spike_coronavirus/pdb_outputs"
output="AgileDesign/results"
motifs="SNNLDSKVGGNYNYR YGFQPTNGVGYQP"
setting_file="setting/setting.yaml"
apptainer exec --nv $sif bash -c "python redesign.py --input_dir $input --output_dir $output --motifs $motifs --setting_file $setting_file"