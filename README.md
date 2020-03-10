# On the Role of Conceptualization in Commonsense Knowledge Graph Construction
[Paper](https://arxiv.org/abs/2003.03239)
# Pretrained models
[Atomic: CCC75-22k](https://hkustconnect-my.sharepoint.com/:u:/g/personal/mhear_connect_ust_hk/Eej13vBzJD1DrNY5BoctdoUBPGFnidIeHZA9ETGaGV6nWw?e=WSdj8j)
[ASER: CCC75-52k](https://hkustconnect-my.sharepoint.com/:u:/g/personal/mhear_connect_ust_hk/EfnMgMtzkjNJnzZtc8pnJZMBOBjkuSDA2GZ4NB12ghpALg?e=sYdNFI)
# Pipeline
## Data
* Probase: https://concept.research.microsoft.com/, to data/probase
* ASER 0.1.0: https://hkust-knowcomp.github.io/ASER/, to data/aser
* Atomic: https://homes.cs.washington.edu/~msap/atomic/, to be decompressed to data/atomic/atomic_data
## Dependencies
```
Unidecode == 1.1.1
inflect == 3.0.2
numpy == 1.16.4
pandas == 0.24.2
spacy == 2.2.3
en-core-web-lg==2.1.0
tensorflow == 1.14.0
torch == 1.3.1
tqdm == 4.36.1
transformers == 2.2.0
Probase-Concept (Available at https://github.com/ScarletPan/probase-concept)
```
## Data preprocess
* collect_probase.py
* aser_build_graph.py
* atomic_build_graph.py
## Training
Sample for running training is given below. Please refer to the code for the meaning of each hyperparameter. Statistics would be available by tensorboard on the log_dir.
* ASER:
```
python train.py
--data_dir=data/aser
--ckpt_dir=ckpt/aser
--log_dir=log/aser
--device='cuda:0'
--hparams='conceptualize_rate=0.75,expr=aser,ht_symmetry=True'
--run_name='aser_ccc75'
--eval_text_path=data/aser/node_contrastive_dev.tsv:data/aser/concept_contrastive_dev.tsv
```
* Atomic
```
python train.py
--data_dir=data/atomic
--ckpt_dir=ckpt/atomic
--log_dir=log/atomic
--device='cuda:0'
--hparams='conceptualize_rate=0.75,expr=atomic,dropout_rate=0.3'
--run_name='atomic_ccc75'
--eval_text_path=data/atomic/node_contrastive_dev.tsv:data/atomic/concept_contrastive_dev.tsv
```
## Generation
* ASER
```
python generate.py
--data_dir=data/aser --run_name=ccc75_100k_gen --device=cuda:0
--output_dir=generations/aser --model_path=ckpt/aser/checkpoint_52000
```
* Atomic
```
python generate.py
--data_dir=data/atomic --run_name=ccc75_140k_gen --device=cuda:0
--output_dir=generations/atomic --model_path=ckpt/atomic/checkpoint_22000
```