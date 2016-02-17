import os

base_dir = '/home/cinjon/Code/dl4mt-material/'
files_dir = os.path.join(base_dir, 'files')

train_src_tok = os.path.join(files_dir, 'train', 'all_en-de.en.tok.shuf')
train_trg_tok = os.path.join(files_dir, 'train', 'all_en-de.de.tok.shuf')

valid_src_tok = os.path.join(files_dir, 'valid', 'newstest-dev-08-10-12.en-de.en.tok.shuf')
valid_trg_tok = os.path.join(files_dir, 'valid', 'newstest-dev-08-10-12.en-de.de.tok.shuf')

dicts_src_tok = os.path.join(files_dir, 'dicts', 'all_en-de.en.tok.shuf.pkl')
dicts_trg_tok = os.path.join(files_dir, 'dicts', 'all_en-de.de.tok.shuf.pkl')
