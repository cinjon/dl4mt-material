import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     maxlen=50,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=5000,
                     dispFreq=250,
                     saveFreq=2500,
                     sampleFreq=1000,
                     use_dropout=params['use-dropout'][0],
                     write_=params['write_'])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': [None],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [50000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True],
        # 'write_': '/home/cinjon/Dropbox/research/dl4mt-material/valid-output'})
        'write_': None})
