# Span-aware pre-trained network with deep information bottleneck for scientific entity relation extraction
PyTorch code for "Span-aware pre-trained network with deep information bottleneck for scientific entity relation extraction". For a description of the model and experiments, see our paper: https://xxx (published at xxx 2024).

![alt text](http://xxx)

## Setup
### Requirements
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard
  - spacy (tested with version 3.0.1) - if installed, used to tokenize sentences for prediction

### Fetch data
Fetch converted (to specific JSON format) SciERC, ADE, BiorelEX, WLP, MECHANIC datasets.

## Examples
(1) Train SciERC on train dataset, evaluate on dev dataset:
```
python ./spert.py train --config configs/scierc_train.conf
```

(2) Evaluate the SciERC model on test dataset:
```
python ./spert.py eval --config configs/scierc_eval.conf
```

## Additional Notes
- To train SpIB with SciBERT download SciBERT from https://github.com/allenai/scibert (under "PyTorch HuggingFace Models") and set "model_path" and "tokenizer_path" in the config file to point to the SciBERT directory. 
- To train SpIB with BioBERT download bioBERT from https://huggingface.co/dmis-lab/biobert-v1.1 (under "PyTorch HuggingFace Models") and set "model_path" and "tokenizer_path" in the config file to point to the BioBERT directory. 
- You can call "python ./spert.py train --help" / "python ./spert.py eval --help" for a description of training/evaluation arguments.

## Acknowledgement 
This task is mainly completed on the basis of Span-based Joint Entity and Relation Extraction with Transformer Pre-training, and we would like to thank the authors very much.
