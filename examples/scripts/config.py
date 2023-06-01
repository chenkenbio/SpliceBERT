
## Users should modify the following paths to their own paths

# NOTE: this script will be used by both python and bash scripts, so please use grammer that is compatible with both languages


## model
SPLICEBERT="/home/chenken/Documents/github/SpliceBERT/models/SpliceBERT.1024nt"
SPLICEBERT_510="/home/chenken/Documents/github/SpliceBERT/models/SpliceBERT.510nt"
SPLICEBERT_HUMAN="/home/chenken/Documents/github/SpliceBERT/models/SpliceBERT-human.510nt"
DNABERT_PREFIX="/home/chenken/Documents/DNABERT/models"

## reference data
hg19="/bigdat1/pub/UCSC/hg19/bigZips/hg19.fa"
hg19_genepred="/home/chenken/Documents/github/SpliceBERT/examples/data/gencode.v41lift37.annotation.genepred.txt.gz"
hg19_phastcons="/bigdat1/pub/UCSC/hg19/hg19.100way.phastCons.h5"
hg19_phylop="/bigdat1/pub/UCSC/hg19/hg19.100way.phyloP100way.h5"
hg38="/home/chenken/db/gencode/GRCh38/GRCh38.primary_assembly.genome.fa"
hg19_regions="/home/chenken/Documents/github/SpliceBERT/examples/data/regions.v41lift37.bed.gz"
hg19_transcript="/home/chenken/Documents/github/SpliceBERT/examples/data/gencode.v41lift37.canonical.tx.bed.gz"

## dataset
VEXSEQ="/home/chenken/Documents/github/SpliceBERT/examples/data/VexSeq.hg19.SNP.vcf"
MFASS="/home/chenken/Documents/github/SpliceBERT/examples/data/MFASS_mmsplice.hg19.vcf"

rand200_transcripts="/home/chenken/Documents/github/SpliceBERT/examples/data/gencode.v41lift37.tx.random-200.bed"
MERCER_DATASET="/home/chenken/Documents/github/SpliceBERT/examples/data/mercer_dataset.hg19.bed.gz"
