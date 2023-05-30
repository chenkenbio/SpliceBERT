# Finetune SpliceBERT to predict Splice sites in the Spliceator dataset

## Extract the dataset

```
$ cd ./data && tar -xzf spliceator.tgz
$ tree 
.
├── spliceator
│   ├── Benchmark
│   │   ├── Danio
│   │   │   ├── SA_sequences_acceptor_400_Final_3.negative.txt
│   │   │   ├── SA_sequences_acceptor_400_Final_3.positive.txt
│   │   │   ├── SA_sequences_donor_400_Final_3.negative.txt
│   │   │   └── SA_sequences_donor_400_Final_3.positive.txt
│   │   ├── Fly
│   │   │   ├── SA_sequences_acceptor_400_Final_3.negative.txt
│   │   │   ├── SA_sequences_acceptor_400_Final_3.positive.txt
│   │   │   ├── SA_sequences_donor_400_Final_3.negative.txt
│   │   │   └── SA_sequences_donor_400_Final_3.positive.txt
│   │   ├── Thaliana
│   │   │   ├── SA_sequences_acceptor_400_Final_3.negative.txt
│   │   │   ├── SA_sequences_acceptor_400_Final_3.positive.txt
│   │   │   ├── SA_sequences_donor_400_Final_3.negative.txt
│   │   │   └── SA_sequences_donor_400_Final_3.positive.txt
│   │   └── Worm
│   │       ├── SA_sequences_acceptor_400_Final_3.negative.txt
│   │       ├── SA_sequences_acceptor_400_Final_3.positive.txt
│   │       ├── SA_sequences_donor_400_Final_3.negative.txt
│   │       └── SA_sequences_donor_400_Final_3.positive.txt
│   └── Training_data
│       ├── Negative
│       │   └── GS
│       │       └── GS_1
│       │           ├── NEG_600_acceptor.csv
│       │           └── NEG_600_donor.csv
│       └── Positive
│           └── GS
│               ├── POS_acceptor_600.csv
│               └── POS_donor_600.csv
└── spliceator.tgz

```

*The complete Spliceator dataset is available at https://git.unistra.fr/nscalzitti/spliceator.*

## Finetune SpliceBERT 

See `run_cv.sh`