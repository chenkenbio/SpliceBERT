# SpliceBERT: precursor messenger RNA langauge model pre-trained on vertebrate pre-mRNAs

SpliceBERT ([manuscript](https://www.biorxiv.org/content/10.1101/2023.01.31.526427v1)) is a pre-mRNA sequence language model pre-trained on over 2 million vertebrate pre-mRNA sequences.
It can be used to study RNA splicing and other biological problems related to pre-mRNA sequence.

![SpliceBERT overview](./overview.png)

- [How to use SpliceBERT?](#how-to-use-splicebert)
- [Reproduce the analysis in manuscript](#reproduce-the-analysis-in-manuscript)
- [Data availability](#Data-availability)  
- [Contact](#contact)
- [Citation](#citation)


## How to use SpliceBERT?

SpliceBERT is implemented with [Huggingface](https://huggingface.co/docs/transformers/index) `transformers` library in PyTorch. Users should install pytorch and transformers to load the SpliceBERT model.  
- Install PyTorch: https://pytorch.org/get-started/locally/  
- Install Huggingface transformers: https://huggingface.co/docs/transformers/installation  

SpliceBERT can be easily used for a series of downstream tasks through the official API.
See [official guide](https://huggingface.co/docs/transformers/model_doc/bert) for more details.

**Download SpliceBERT**

The weights of SpliceBERT can be downloaded from [zenodo](https://doi.org/10.5281/zenodo.7995778): https://zenodo.org/record/7995778/files/models.tar.gz?download=1

**System requirements**  

We recommend running SpliceBERT on a Linux system with a NVIDIA GPU of at least 4GB memory. (Running our model with only CPU is possible, but it will very slow.)

**Examples**:
```python
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification

SPLICEBERT_PATH = "/path/to/SpliceBERT/model"  # set the path to the folder of pre-trained SpliceBERT

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)

# prepare input sequence
seq = "ACGUACGuacguaCGu"  ## WARNING: this is just a demo. SpliceBERT may not work on sequences shorter than 64nt as it was trained on sequences of 64-1024nt in length
seq = ' '.join(list(seq.upper().replace("U", "T"))) # U -> T and add whitespace
input_ids = tokenizer.encode(seq) # N -> 5, A -> 6, C -> 7, G -> 8, T(U) -> 9. warning: a [CLS] and a [SEP] token will be added to the start and the end of seq
input_ids = torch.as_tensor(input_ids) # convert python list to Tensor
input_ids = input_ids.unsqueeze(0) # add batch dimension, shape: (batch_size, sequence_length)


# get nucleotide embeddings (hidden states)
model = AutoModel.from_pretrained(SPLICEBERT_PATH) # load model
last_hidden_state = model(input_ids).last_hidden_state # get hidden states from last layer
hiddens_states = model(input_ids, output_hidden_states=True).hidden_states # hidden states from the embedding layer (nn.Embedding) and the 6 transformer encoder layers

# get nucleotide type logits in masked language modeling
model = AutoModelForMaskedLM.from_pretrained(SPLICEBERT_PATH) # load model
logits = model(input_ids).logits # shape: (batch_size, sequence_length, vocab_size)

# finetuning SpliceBERT for token classification tasks
model = AutoModelForTokenClassification.from_pretrained(SPLICEBERT_PATH, num_labels=3) # assume the class number is 3, shape: (batch_size, sequence_length, num_labels)

# finetuning SpliceBERT for sequence classification tasks
model = AutoModelForSequenceClassification.from_pretrained(SPLICEBERT_PATH, num_labels=3) # assume the class number is 3, shape: (batch_size, sequence_length, num_labels)

```

## Reproduce the analysis in manuscript  

Before running the codes, run `bash download.sh` to fetch the data used in the analysis.
Then, run `bash setup.sh` in the `./examples` folder to compile the codes written in cython (`cython` is required).  

Pre-computed results of section 1-4 are available at [Google Drive](https://drive.google.com/file/d/1TCwbhyMiBP1bGEQcZj1qTH-Rif7Ee93P/view?usp=sharing). Users can download the results to `./examples` and decompress it to skip the step of running bash script `pipeline.sh` in jupyter notebooks.

The codes for analyzing SpliceBERT are available in [examples](./examples):  
1. [evolutionary conservation analysis](./examples/00-conservation) (related to Figure 1)  
2. [nucleotide embedding analysis](./examples/02-embedding) (related to Figure 2)  
3. [attention weight analysis](./examples/03-attention) (related to Figure 3)  
4. [variant effect analysis](./examples/01-variant) (related to Figure 4)  
5. [branchpoint prediction](./examples/05-bp-prediction) (related to Figure 5)
6. [splice site prediction](./examples/04-splicesite-prediction) (related to Figure 6)


We generated the results in this manuscript under the following environment:
- System:
	- Ubuntu 20.04 LTS
- Python packages:
	- `Python (3.9.7)`
	- `transformers (4.24.0)`  
	- `pytorch (1.12.1)`  
	- `h5py (3.2.1)`
	- `numpy (1.23.3)`  
	- `scipy (1.8.0)`  
	- `scikit-learn (1.1.1)`  
	- `scanpy (1.8.2)`
	- `matplotlib (3.5.1)`  
	- `seaborn (0.11.2)`
	- `tqdm (4.64.0)`  
	- `pyBigWig (0.3.18)`
	- `cython (0.29.28)`
- Command line tools:  
	- `bedtools (2.30.0)`  
	- `MaxEntScan (2004)`
	- `gtfToGenePred (v377)`



*Note: the version number is only used to illustrate the version of softwares used in our study. In most cases, users do not need to ensure that the versions are strictly the same to ours to run the codes*

## Data availability
The model weights and data for running the scripts are available at [zenodo](https://doi.org/10.5281/zenodo.7995778).

## Contact
For issues related to the scripts, create an issue at https://github.com/biomed-AI/SpliceBERT/issues.

For any other questions, feel free to contact chenkenbio {at} gmail.com.

## Citation

```TeX
@article{Chen2023.01.31.526427,
	author = {Chen, Ken and Zhou, Yue and Ding, Maolin and Wang, Yu and Ren, Zhixiang and Yang, Yuedong},
	title = {Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction},
	year = {2023},
	doi = {10.1101/2023.01.31.526427},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/02/03/2023.01.31.526427},
	journal = {bioRxiv}
}
```
