import os

base_dir = '/home/ubuntu/research/dl4mt-material/'

train_en_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.en.tok.shuf')
train_fr_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.fr.tok.shuf')

valid_en_tok = os.path.join(base_dir, 'files', 'newstest2013.en.tok.shuf')
valid_fr_tok = os.path.join(base_dir, 'files', 'newstest2013.fr.tok.shuf')

dicts_en_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.en.tok.shuf.pkl')
dicts_fr_tok = os.path.join(base_dir, 'files', 'europarl-v7.fr-en.fr.tok.shuf.pkl')
