import os, sys, random, time, datetime

random.seed(0)
import numpy as np

np.random.seed(0)
import torch

torch.manual_seed(0)

import argparse
from hparams import hparams, hparams_debug_string
import logging
from tqdm import tqdm
from datafeeder import DataFeeder, EdgeSample, SampleType, FullGenerationFeeder
import pickle

from transformers import BertTokenizer
from model import EdgeClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_step_shift = 0

from train import run_eval

def run(args):
    output_dir = os.path.join(args.output_dir, args.run_name + '_' + datetime.datetime.now().strftime('%m%d_%H%M'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fmt = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    h = logging.FileHandler(os.path.join(output_dir, 'outputs.log'))
    h.setFormatter(fmt)
    h_ = logging.StreamHandler(sys.stdout)
    h_.setFormatter(fmt)
    logging.basicConfig(handlers=[h, h_], level=logging.INFO)
    if args.device is not None:
        global device
        logging.info('Setting device to ' + args.device)
        device = torch.device(args.device)

    logging.info('Setting up...')
    hparams.parse(args.hparams)
    logging.info(hparams_debug_string())
    logging.info('Load saved model from %s ...' % (args.model_path))
    model = EdgeClassification
    model = model.from_pretrained(args.model_path, hparams=hparams)
    step = args.model_path.split('_')[-1]
    if step.isnumeric():
        global current_step_shift
        current_step_shift = int(step)
        logging.info('Initial step=%d' % current_step_shift)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    gen_feeder = FullGenerationFeeder(args.data_dir, hparams, tokenizer, 'test')

    model.to(device)
    model.eval()

    logging.info('Start training...')

    if args.n_generate is None:
        n_generate = 1e9
        print('Infinite generations')
    else:
        n_generate = int(args.n_generate)
        print('Generate %d' % n_generate)

    collected = []
    thrs = args.score_threshold
    n_accepted = 0

    while len(collected) < n_generate:
        batch = gen_feeder.next_batch()
        if batch is None:
            break
        samples = gen_feeder._last_samples
        base_samples = gen_feeder._base_sample_ids
        with torch.no_grad():
            outputs = model(input_ids=batch.input_ids.to(device),
                            attention_mask=batch.input_mask.to(device),
                            token_type_ids=batch.token_type_ids.to(device),
                            labels=batch.labels.to(device))
            probs = outputs['probs'].detach().cpu().numpy()[:, 1] # prob of "positive"
        for i in range(len(samples)):
            collected.append((base_samples[i], samples[i], probs[i]))
        n_accepted += np.sum(probs > thrs)
        print('[%d/%d] %d collected, %d accepted, moving rate=%.2f' % (
            gen_feeder._offset, len(gen_feeder._indices), len(collected), n_accepted, n_accepted / len(collected)))

    pickle.dump(collected, open(os.path.join(output_dir, 'generations.pickle'), 'wb'))
    pickle.dump(gen_feeder._proposed_samples, open(os.path.join(output_dir, 'proposed_samples.pickle'), 'wb'))
    with open(os.path.join(output_dir, 'generations.txt'), 'w') as fw:
        for bs, sample, prob in collected:
            if prob > thrs:
                fw.write('%s\t%s\t%s\t%.4f\n' % (sample[0], sample[1], sample[2], prob))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--model_path', default=None, type=str, required=True,
                        help="Path for saved model")
    parser.add_argument('--device', default=None, type=str,
                        help="Device to run on")
    parser.add_argument('--run_name', default='', type=str,
                        help="Name of the run, for as suffix of output_dir")
    parser.add_argument('--n_generate', default=None, type=int,
                        help='Number of generated samples')
    parser.add_argument('--score_threshold', default=0.5, type=float,
                        help='Threshold of score/prob')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    run(args)
