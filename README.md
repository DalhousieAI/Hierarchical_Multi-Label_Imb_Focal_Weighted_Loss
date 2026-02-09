## Introduction

The purpose of this repo is to provied code support for the paper "[Improving Detection of Rare Nodes in Hierarchical Multi-Label Learning](https://openreview.net/forum?id=hf4zEWWIvE)".
This work is on a problem commonly encountered in hierarchical multi-label (HML) classification, where due to naturally arising source of imabalnce in collected data, deeper, specific nodes are difficult to recall by trained networks.
This issue can be particularly problematic because the rare nodes are often indicative of the most valued entity - either for research, or in terms of economics.
We approach this problem by implemented an inverse frequency term from a node-wise perspective rather than a typical observation-based approach.
We also introduce an uncertainty based term, resembling that in focal loss[1], using recent uncertainty quantification research. 
Together, these terms added to HML classification loss[2] and prioritize rare and uncertain nodes during training, improving their recall.
We find that this weighted loss approach can provide significant boosts to recall and F1 over existing HML imbalance techniques in difficult classification scenarios.

## Installation

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Contents
The main scripts are:

- `build_and_run_focal_model.py` → script for running model with imbalance and focal weighting
- `build_and_run_model.py` → script for running with a resampling method or imbalance weighting only
- `echinoderms_run_focal_model.py` → script for running vision model with imbalance and focal weighting
- `echinoderms_run_model.py` → script for running vision model with only imbalance weights

## References

[1] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.  
*Focal Loss for Dense Object Detection*.  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2):318–327, 2020.  
doi: 10.1109/TPAMI.2018.2858826

[2] Eleonora Giunchiglia and Thomas Lukasiewicz.  
*Coherent Hierarchical Multi-Label Classification Networks*.  
In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.),  
Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9662–9673, 2020.  
https://proceedings.neurips.cc/paper_files/paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf
