# Cathode Screening

A machine learning system for high-throughput screening of lithium-ion battery cathode materials. The system predicts thermodynamic stability from crystal structure using graph neural networks with calibrated uncertainty quantification.

## Problem

Discovering new cathode materials requires evaluating thermodynamic stability through Density Functional Theory (DFT) calculations. These calculations are computationally expensive, often taking hours per structure. With thousands of candidate materials, exhaustive DFT screening becomes prohibitively costly.

This project reduces that computational burden by providing fast, reliable stability predictions that identify promising candidates before committing to full DFT evaluation.

## Approach

The pipeline ingests crystal structures from the Materials Project database and applies domain-specific filters for cathode-relevant chemistries (lithium-oxygen systems with transition metals). Each structure is converted to a periodic graph representation where atoms are nodes and interatomic distances define edges.

A Crystal Graph Convolutional Neural Network (CGCNN) processes these graphs to predict energy above hull (E_hull), the primary metric for thermodynamic stability. The model outputs three quantities:

1. Predicted E_hull value
2. Learned uncertainty estimate (heteroscedastic)
3. Probability of stability (E_hull < 0.05 eV)

Training uses chemistry-aware data splits to prevent information leakage between structurally similar compounds. Post-hoc calibration ensures uncertainty estimates are reliable for downstream decision-making.

## Decision Framework

The outputs support a risk-controlled screening policy:

- Candidates with high stability probability and low uncertainty proceed to DFT validation
- Candidates with mixed signals are flagged for further analysis
- Candidates with low stability probability are deprioritized

This framework maximizes discovery rate while controlling false positives, reducing wasted DFT cycles on unstable materials.

## Technical Stack

- Crystal graph representation with radial basis function edge encoding
- 4-layer message passing network with attention pooling
- Huber loss for robust regression under heavy-tailed error distributions
- Conformal prediction for coverage-guaranteed prediction intervals

## License

Proprietary. All rights reserved.

