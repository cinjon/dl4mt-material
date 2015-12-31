#!/bin/bash -x
# This script sets up development and data environments for
# a local machine, copy under your home directory and run.
# Note that, Theano is NOT installed by this script.

# code directory for cloned repositories
CODE_DIR=/home/ubuntu/research/dl4mt-material

# code repository
CODE_CENTRAL=https://github.com/cinjon/dl4mt-material

# our data manipulation files will reside here
DATA_DIR=${CODE_DIR}/data

# our training files will reside here
FILES_DIR=${CODE_DIR}/files

# our trained models will be saved here
MODELS_DIR=${CODE_DIR}/models

# clone the repository from github into code directory
if [ ! -d "${CODE_DIR}" ]; then
    mkdir -p ${CODE_DIR}
    git clone ${CODE_CENTRAL} ${CODE_DIR}
fi

# download the europarl v7 and validation sets and extract
python ${CODE_DIR}/data/download_files.py \
    -s='fr' -t='en' \
    --source-dev=newstest2013.fr \
    --target-dev=newstest2013.en \
    --outdir=${FILES_DIR}

# tokenize corresponding files
perl ${CODE_DIR}/data/tokenizer.perl -l 'fr' < ${FILES_DIR}/test/newstest2013.fr > ${FILES_DIR}/newstest2013.fr.tok
perl ${CODE_DIR}/data/tokenizer.perl -l 'en' < ${FILES_DIR}/test/newstest2013.en > ${FILES_DIR}/newstest2013.en.tok
perl ${CODE_DIR}/data/tokenizer.perl -l 'fr' < ${FILES_DIR}/europarl-v7.fr-en.fr > ${FILES_DIR}/europarl-v7.fr-en.fr.tok
perl ${CODE_DIR}/data/tokenizer.perl -l 'en' < ${FILES_DIR}/europarl-v7.fr-en.en > ${FILES_DIR}/europarl-v7.fr-en.en.tok

# extract dictionaries
python ${CODE_DIR}/data/build_dictionary.py ${FILES_DIR}/europarl-v7.fr-en.fr.tok
python ${CODE_DIR}/data/build_dictionary.py ${FILES_DIR}/europarl-v7.fr-en.en.tok

# create model output directory if it does not exist
if [ ! -d "${MODELS_DIR}" ]; then
    mkdir -p ${MODELS_DIR}
fi

# check if theano is working
python -c "import theano;print 'theano available!'"
