import os

base_dir = '~/Code/research/dl4mt-cinjon/'

train_en_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.en.tok')
train_fr_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.fr.tok')

valid_en_tok = os.path.join(base_dir, 'files', 'newstest2013.en.tok')
valid_fr_tok = os.path.join(base_dir, 'files', 'newstest2013.fr.tok')

dicts_en_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.en.tok.pkl')
dicts_fr_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.en.tok.pkl')
