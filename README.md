# Prompt-based distribution alignment
Prompt-based Distribution Alignment for Domain Generalization in Text Classification.

It is released with our EMNLP 2022 paper: [Prompt-based Distribution Alignment for Domain Generalization in Text Classification](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.690/). This repo contains the code in our paper. 


### Introduction
The code is built based on the open-source toolkit [OpenPrompt](https://github.com/thunlp/OpenPrompt). 

### Requirements
```
python >= 3.7
torch >= 1.10.0
transformers >= 4.10.0
```

### Usage
```
python experiments/cli.py --config_yaml classification_manual_prompt.yaml 
```

  


### Citation
When you use the our paper or dataset, we would appreciate it if you cite the following:
```
@inproceedings{jia2022pda,
	title={Prompt-based distribution alignment for domain generalization in text classification},
	author={Jia, Chen and Zhang, Yue},
	booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
	pages={10147–-10157},
	year={2022}
}

```
