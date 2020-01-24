import os, random, time, datetime

random.seed(0)
import numpy as np

np.random.seed(0)
import torch

torch.manual_seed(0)

import argparse
from hparams import hparams, hparams_debug_string
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
from datafeeder import DataFeeder, ExternalTextFeeder, PrebuiltTrainFeeder
from utils import ValueWindow, set_logger

from transformers import BertTokenizer, AdamW, BertConfig, RobertaTokenizer, RobertaConfig
from model import EdgeClassification, RobertaEdgeClassification

device = None

def evaluate(args, model, feeder, hparams):
    all_preds = []
    losses = []
    model.eval()
    all_labels = []
    with torch.no_grad():
        while True:
            batch = feeder.next_batch()
            if batch is None:
                break
            outputs = model(input_ids=batch.input_ids.to(device),
                            attention_mask=batch.input_mask.to(device),
                            token_type_ids=None if batch.token_type_ids is None else batch.token_type_ids.to(device),
                            labels=batch.labels.to(device))
            losses.append(outputs['loss'].item())
            all_preds.append(outputs['preds'].cpu().detach().numpy())
            all_labels.append(batch.labels.detach().numpy())
    print('Finished eval...')
    losses = np.mean(losses)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = np.sum(all_preds == all_labels) / len(all_preds)
    t_acc = np.sum(np.logical_and(all_preds == 1, all_labels == 1)) / np.sum(all_labels == 1)
    f_acc = np.sum(np.logical_and(all_preds == 0, all_labels == 0)) / np.sum(all_labels == 0)
    binary_acc = np.sum((all_preds == 1) == (all_labels == 1)) / len(all_preds)
    return {'loss': losses, 'acc': acc, 'binary_acc': binary_acc, 't_acc': t_acc, 'f_acc': f_acc}


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, lr_decay_step, max_lr_decay_rate):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(max_lr_decay_rate, float(lr_decay_step - current_step)
                   / float(max(1, lr_decay_step - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def run_eval(args, model, eval_feeder):
    logs = {}
    start_time = time.time()
    for name, e in eval_feeder:
        results = evaluate(args, model, e, hparams)
        if name:
            key_name = name + '/'
        else:
            key_name = ''
        for key, value in results.items():
            eval_key = 'eval/' + key_name + key
            logs[eval_key] = value
        step_time = time.time() - start_time
        result_msg = ' '.join(['%s=%.04f' % (k, v) for k, v in logs.items()])
        message = 'Evaluation in %.02f sec: %s' % (step_time, result_msg)
        if name:
            message = '[%s] ' % name + message
        logging.info(message)
    return logs

def train(args):
    if args.model_path is None:
        msg = 'Prepare for new run ...'
        output_dir = os.path.join(args.log_dir, args.run_name + '_' + datetime.datetime.now().strftime('%m%d_%H%M'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ckpt_dir = os.path.join(args.ckpt_dir, args.run_name + '_' + datetime.datetime.now().strftime('%m%d_%H%M'))
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    else:
        msg = 'Restart previous run ...\nlogs to save to %s, ckpt to save to %s, model to load from %s' % \
                     (args.log_dir, args.ckpt_dir, args.model_path)
        output_dir = args.log_dir
        ckpt_dir = args.ckpt_dir
        if not os.path.isdir(output_dir):
            print('Invalid log dir: %s' % output_dir)
            return
        if not os.path.isdir(ckpt_dir):
            print('Invalid ckpt dir: %s' % ckpt_dir)
            return

    set_logger(os.path.join(output_dir, 'outputs.log'))
    logging.info(msg)

    global device
    if args.device is not None:
        logging.info('Setting device to ' + args.device)
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info('Setting up...')
    hparams.parse(args.hparams)
    logging.info(hparams_debug_string())

    model = EdgeClassification

    if hparams.use_roberta:
        logging.info('Using Roberta...')
        model = RobertaEdgeClassification

    global_step = 0

    if args.model_path is None:
        if hparams.load_pretrained:
            logging.info('Load online pretrained model...' +
                         (('cached at ' + args.cache_path) if args.cache_path is not None else ''))
            if hparams.use_roberta:
                model = model.from_pretrained('roberta-base', cache_dir=args.cache_path, hparams=hparams)
            else:
                model = model.from_pretrained('bert-base-uncased', cache_dir=args.cache_path, hparams=hparams)
        else:
            logging.info('Build model from scratch...')
            if hparams.use_roberta:
                config = BertConfig.from_pretrained('bert-base-uncased')
            else:
                config = RobertaConfig.from_pretrained('roberta-base')
            model = model(config=config, hparams=hparams)
    else:
        if not os.path.isdir(args.model_path):
            raise OSError(str(args.model_path) + ' not found')
        logging.info('Load saved model from %s ...' % (args.model_path))
        model = model.from_pretrained(args.model_path, hparams=hparams)
        step = args.model_path.split('_')[-1]
        if step.isnumeric():
            global_step = int(step)
            logging.info('Initial step=%d' % global_step)

    if hparams.use_roberta:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    hparams.parse(args.hparams)
    logging.info(hparams_debug_string())

    if hparams.text_sample_eval:
        if args.eval_text_path is None:
            raise ValueError('eval_text_path not given')
        if ':' not in args.eval_text_path:
            eval_data_paths = [args.eval_text_path]
        else:
            eval_data_paths = args.eval_text_path.split(':')
        eval_feeder = []
        for p in eval_data_paths:
            name = os.path.split(p)[-1]
            if name.endswith('.tsv'):
                name = name[:-4]
            eval_feeder.append((name, ExternalTextFeeder(p, hparams, tokenizer, 'dev')))
    else:
        eval_feeder = [('', DataFeeder(args.data_dir, hparams, tokenizer, 'dev'))]

    tb_writer = SummaryWriter(output_dir)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': hparams.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=hparams.learning_rate, eps=hparams.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=hparams.warmup_steps,
                                                lr_decay_step=hparams.lr_decay_step,
                                                max_lr_decay_rate=hparams.max_lr_decay_rate)

    acc_step = global_step * hparams.gradient_accumulation_steps
    time_window = ValueWindow()
    loss_window = ValueWindow()
    acc_window = ValueWindow()
    model.to(device)
    model.zero_grad()
    tr_loss = tr_acc = 0.0
    start_time = time.time()

    if args.model_path is not None:
        logging.info('Load saved model from %s ...' % (args.model_path))
        if os.path.exists(os.path.join(args.model_path, 'optimizer.pt')) \
                and os.path.exists(os.path.join(args.model_path, 'scheduler.pt')):
            optimizer.load_state_dict(torch.load(os.path.join(args.model_path, 'optimizer.pt')))
            optimizer.load_state_dict(optimizer.state_dict())
            scheduler.load_state_dict(torch.load(os.path.join(args.model_path, 'scheduler.pt')))
            scheduler.load_state_dict(scheduler.state_dict())
        else:
            logging.warning('Could not find saved optimizer/scheduler')

    if global_step > 0:
        logs = run_eval(args, model, eval_feeder)
        for key, value in logs.items():
            tb_writer.add_scalar(key, value, global_step)

    logging.info('Start training...')
    if hparams.text_sample_train:
        train_feeder = PrebuiltTrainFeeder(args.train_text_path, hparams, tokenizer, 'train')
    else:
        train_feeder = DataFeeder(args.data_dir, hparams, tokenizer, 'train')

    while True:
        batch = train_feeder.next_batch()
        model.train()

        outputs = model(input_ids=batch.input_ids.to(device),
                        attention_mask=batch.input_mask.to(device),
                        token_type_ids=None if batch.token_type_ids is None else batch.token_type_ids.to(device),
                        labels=batch.labels.to(device))
        loss = outputs['loss']
        preds = outputs['preds']

        acc = torch.mean((preds.cpu() == batch.labels).float())
        preds = preds.cpu().detach().numpy()
        labels = batch.labels.detach().numpy()
        t_acc = np.sum(np.logical_and(preds == 1, labels == 1)) / np.sum(labels == 1)
        f_acc = np.sum(np.logical_and(preds == 0, labels == 0)) / np.sum(labels == 0)

        if hparams.gradient_accumulation_steps > 1:
            loss = loss / hparams.gradient_accumulation_steps
            acc = acc / hparams.gradient_accumulation_steps

        tr_loss += loss.item()
        tr_acc += acc.item()
        loss.backward()
        acc_step += 1

        if acc_step % hparams.gradient_accumulation_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)
        optimizer.step()
        scheduler.step(None)
        model.zero_grad()
        global_step += 1

        step_time = time.time() - start_time
        time_window.append(step_time)
        loss_window.append(tr_loss)
        acc_window.append(tr_acc)

        if global_step % args.save_steps == 0:
            # Save model checkpoint
            model_to_save = model.module if hasattr(model, 'module') else model
            cur_ckpt_dir = os.path.join(ckpt_dir, 'checkpoint_%d' % (global_step))
            if not os.path.exists(cur_ckpt_dir):
                os.makedirs(cur_ckpt_dir)
            model_to_save.save_pretrained(cur_ckpt_dir)
            torch.save(args, os.path.join(cur_ckpt_dir, 'training_args.bin'))
            torch.save(optimizer.state_dict(), os.path.join(cur_ckpt_dir, 'optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(cur_ckpt_dir, 'scheduler.pt'))
            logging.info("Saving model checkpoint to %s", cur_ckpt_dir)

        if global_step % args.logging_steps == 0:
            logs = run_eval(args, model, eval_feeder)

            learning_rate_scalar = scheduler.get_lr()[0]
            logs['learning_rate'] = learning_rate_scalar
            logs['loss'] = loss_window.average
            logs['acc'] = acc_window.average

            for key, value in logs.items():
                tb_writer.add_scalar(key, value, global_step)

        message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, acc=%.05f, avg_acc=%.05f, t_acc=%.05f, f_acc=%.05f]' % (
            global_step, step_time, tr_loss, loss_window.average, tr_acc, acc_window.average, t_acc, f_acc)
        logging.info(message)
        tr_loss = tr_acc = 0.0
        start_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ckpt_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--log_dir", default=None, type=str, required=True,
                        help="The output directory where logs & tensorboards will be written.")
    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--cache_path', default=None, type=str,
                        help="Path for downloaded pretrained model.")
    parser.add_argument('--model_path', default=None, type=str,
                        help="Path for saved model to be re-loaded.")
    parser.add_argument('--device', default=None, type=str,
                        help="Device to run on")
    parser.add_argument('--run_name', default='', type=str,
                        help="Name of the run, for as suffix of output_dir")
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--train_text_path', default=None,
                        help='Text file for train, performed by PrebuiltTrainFeeder')
    parser.add_argument('--eval_text_path', default=None,
                        help='Text file for eval, performed by ExternalTextFeeder')
    args = parser.parse_args()
    train(args)
