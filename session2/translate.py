'''
Translates a source file using a translation model.
'''
import argparse
import datetime

import numpy
import cPickle as pkl

from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue


def translate_model(queue, rqueue, pid, model, options, k, normalize, annotations_only):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    # f_init outs are [init_state (to decoder), ctx (from encoder)]
    # f_next outs are [next_probs, next_sample, next_state] (decoder)
    f_init, f_next = build_sampler(tparams, options, trng, annotations_only)

    def _translate(seq):
        # sample given an input sequence and obtain scores
        if annotations_only:
            next_state, ctx = f_init(numpy.array(seq).reshape([len(seq), 1]))
            return ctx
        else:
            sample, score = gen_sample(tparams, f_init, f_next,
                                       numpy.array(seq).reshape([len(seq), 1]),
                                       options, trng=trng, k=k, maxlen=200,
                                       stochastic=False, argmax=False)

            # normalize scores according to sequence lengths
            if normalize:
                lengths = numpy.array([len(s) for s in sample])
                score = score / lengths
            sidx = numpy.argmin(score)
            return sample[sidx]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        seq = _translate(x)

        rqueue.put((idx, seq))

    return


def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, n_process=5, chr_level=False,
         annotations_only=False, max_lines=None):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, model, options, k, normalize, annotations_only))
        processes[midx].start()

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if max_lines and idx == max_lines:
                    print 'Breaking because we reached the max_lines'
                    return idx
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
                x += [0]
                queue.put((idx, x))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        for idx in xrange(n_samples):
            ridx, rseq = rqueue.get()
            trans[ridx] = rseq
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans

    n_samples = _send_jobs(source_file)
    trans = _retrieve_jobs(n_samples)
    if not annotations_only:
        print 'Running seqs2words because annotations_only is false.'
        trans = _seqs2words(trans)
    _finish_processes()
    print 'Writing trans.'
    if annotations_only:
        numpy.set_printoptions(threshold=numpy.nan)
        trans = numpy.array(trans)
        numpy.savez(open(saveto + '.npz', 'w'), trans)
    else:
        with open(saveto, 'w') as f:
            print 'Printing translations to', saveto
            print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-c', action="store_true", default=False)
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)
    # get just the embedding annotations and save them to "saveto"
    parser.add_argument('-annotations', action="store_true", default=False)
    parser.add_argument('-max_lines', type=int, default=0)

    args = parser.parse_args()
    print args

    start = datetime.datetime.utcnow()
    print 'Started at ', start
    main(args.model, args.dictionary, args.dictionary_target, args.source,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p,
         chr_level=args.c, annotations_only=args.annotations, max_lines=args.max_lines)
    end = datetime.datetime.utcnow()
    print 'Time to finish -->', end - start
