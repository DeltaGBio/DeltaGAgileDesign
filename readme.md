# DeltaGAgileDesign

## Computational Pipeline for Antigen Motif Inpainting and Redesign

![RFdiffusion workflow diagram](image/RF_diffusion.png)

This repository contains a pipeline for designing and optimizing protein antigens through a three-stage process:

1. **RFdiffusion** - Inpainting of antigen motifs
2. **MPNN** - Sequence redesign
3. **AlphaFold** - Structure prediction

## Overview

The pipeline enables the design of novel protein antigens by identifying and optimizing critical motifs. 

## Pipeline EGFR Case Study

As a proof of concept, we apply our pipeline to redesign the Epidermal Growth Factor Receptor (EGFR) while preserving its critical interface interactions. EGFR is a transmembrane receptor tyrosine kinase involved in cell signaling pathways that regulate proliferation, differentiation, and survival.

### Workflow Steps

1. **Structure Preparation**
   - Extract the EGFR extracellular domain (PDB: 4UV7) and its binding interface with EGF.
   - Clean and prepare the structure for computational design.

2. **Interface Motif Definition**
   - Identify key residues at the EGFR-antibody interface using distance-based criteria.
   - Define the geometry of the binding motif to be preserved during design.

3. **RFdiffusion Inpainting**
   - Constrain the key interface residues as a motif.
   - Generate novel scaffold designs that accommodate the EGFR binding motif.
   - Sample diverse structural solutions (n = 1000). 

4. **MPNN Sequence Design**
   - Apply ProteinMPNN to optimize sequences for each scaffold.
   - Constrain interface residues to maintain binding function.
   - Generate 25 sequence variants per scaffold.

5. **AlphaFold Validation**
   - Predict the structures of the designed sequences.
   - Evaluate pLDDT scores for overall confidence.
   - Focus on the stability of the interface region and the predicted binding geometry.

## Installation

To be updated

## Citation

If you use this pipeline in your research, please cite the following papers:

- RFdiffusion: [Watson et al., Science (2023)](https://www.nature.com/articles/s41586-023-06415-8)
- ProteinMPNN: [Dauparas et al., Science (2022)](https://www.science.org/doi/10.1126/science.add2187)
- AlphaFold: [Jumper et al., Nature (2021)](https://www.nature.com/articles/s41586-021-03819-2)

## License

This project is licensed under the MIT License - see the LICENSE file for details.


