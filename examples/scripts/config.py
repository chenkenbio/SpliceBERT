
## Users can modify the following paths to their own paths

# NOTE: this script will be used by both python and bash scripts, so please use grammer that is compatible with both languages

## model
SPLICEBERT="../../models/SpliceBERT.1024nt"
SPLICEBERT_510="../../models/SpliceBERT.510nt"
SPLICEBERT_HUMAN="../../models/SpliceBERT-human.510nt"
DNABERT_PREFIX="../../models/dnabert"

## reference data
hg19="../data/hg19.fa"
hg19_genepred="../data/gencode.v41lift37.annotation.genepred.txt.gz"
hg19_phastcons="../data/hg19.100way.phastCons.h5"
hg19_phylop="../data/hg19.100way.phyloP100way.h5"
hg38="../data/GRCh38.primary_assembly.genome.fa"
hg19_regions="../data/regions.v41lift37.bed.gz"
hg19_transcript="../data/gencode.v41lift37.canonical.tx.bed.gz"

## dataset
VEXSEQ="../data/VexSeq.hg19.SNP.vcf.gz"
MFASS="../data/MFASS_mmsplice.hg19.vcf.gz"

rand200_transcripts="../data/gencode.v41lift37.tx.random-200.bed.gz"
MERCER_DATASET="../data/mercer_dataset.hg19.bed.gz"
