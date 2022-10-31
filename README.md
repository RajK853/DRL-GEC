# DRL-GEC
Grammar Error Correction via Deep Reinforcement Learning

# Setup

## Conda environment
```commandline
git clone --recurse-submodules git@github.com:RajK853/DRL-GEC.git
cd DRL-GEC
conda env create -f environment.yml
```

## Install Ipykernel for Jupyter Notebook
```commandline
conda activate drl-gec
python -m ipykernel install --user --name=drl-gec
```

# Remove hindi sentences from Lang-8 training dataset

Lang-8 training dataset (`lang8.train.auto.bea19.m2`) has some hindi sentences in between the lines `1,193,666` and `1,193,828`.
```
...
S Woh kaun hai ?
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

S - Who is that ?
A 3 4|||R:PRON|||he|||REQUIRED|||-NONE-|||0

S Yeh kiska ghar hai ?
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

S - Whose is this house ?
A 2 5|||R:WO|||house is this|||REQUIRED|||-NONE-|||0

S Daftar mein kaun hai ? - Who is in the office ?
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
...
```
The `fix_lang8.sh` bash script removes these lines from the original M2 training file using `sed`.
```bash
./fix_lang8.sh path/to/lang8/lang8.train.auto.bea19.m2 path/to/lang8/lang8.train.auto.bea19.cleaned.m2
```