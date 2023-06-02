# SpliceBERT: model weights

Model weights can be downloaded from zenodo: https://zenodo.org/record/7995778

- SpliceBERT.1024nt: pre-trained on vertebrate mRNA sequence fragments from 64nt to 1024nt (variable length)  
- SpliceBERT.510nt: pre-trained on vertebrate mRNA sequence fragments of 510nt (fixed length)  
- SpliceBERT-human.510nt: pre-trained on human mRNA sequence fragments of 510nt (fixed length)  

**WARNING**: `SpliceBERT.510nt` and `SpliceBERT-human.510nt` may not work properly on sequences whose length is not 510nt (excluding [CLS] and [SEP]) without finetuning.

