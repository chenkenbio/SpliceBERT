# SpliceBERT

SpliceBERT is a nucleotide sequence langauge model pre-trained on more than 2 million mRNA sequences for studying RNA splicing. 

## Quickstart

```python
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./SpliceBERT")

# prepare input sequence
seq = "ACGTACGtacgtaCGt"  ## WARNING: this is just a demo. SpliceBERT may not work on sequences shorter than 64nt as it was trained on sequences of 64-1024nt in length
seq = ' '.join(list(seq.upper().replace("U", "T"))) # add whitespace
input_ids = tokenizer.encode(seq) # warning: a [CLS] and a [SEP] token will be added to the start and the end of seq
input_ids = torch.as_tensor(input_ids)
input_ids = input_ids.unsqueeze(0) # add batch dimension


# get hidden states (nucleotide embeddings)
model = AutoModel.from_pretrained("./SpliceBERT") # load model
last_hidden_state = model(input_ids).last_hidden_state

# get logits from MLM
model = AutoModelForMaskedLM.from_pretrained("./SpliceBERT") # load model
logits = model(input_ids).logits


# load pre-trained SpliceBERT for token classification
model = AutoModelForTokenClassification.from_pretrained("./SpliceBERT/", num_labels=3) # assume the number of classes is 3

# load pre-trained SpliceBERT for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("./SpliceBERT/", num_labels=3) # assume the number of classes is 3

```




