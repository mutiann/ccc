import torch
import os, random, traceback
from typing import NamedTuple, List, Optional, Tuple, Union
from utils import Node, Edge, EntityMention, Skeleton, label_to_text
from conceptualize_proposer import Proposer, Substitution
import json
import tqdm
from transformers import tokenization_bert, RobertaTokenizer
import logging
from collections import defaultdict
import numpy as np
from enum import IntEnum


class SampleType(IntEnum):
    CC = 0
    TRUTH = 1
    NODE = 2
    NAT = 3


class Example(NamedTuple):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    labels: torch.Tensor
    token_type_ids: torch.Tensor

class EdgeSample(NamedTuple):
    head_text: str
    edge: str
    tail_text: str
    type: SampleType

    def __str__(self):
        return '\t'.join([self.head_text, self.edge, self.tail_text, str(self.type)])

atomic_edges = ['xIntent', 'xNeed', 'xAttr', 'xReact', 'xWant', 'xEffect', 'oReact', 'oWant', 'oEffect']
atomic_edges = dict(zip(atomic_edges, range(len(atomic_edges))))

exclude_text_samples = set() # (u_text, type, v_text)

def build_batch_from_edge(tokenizer, max_seq_length, samples: List[EdgeSample]) -> Example:
    inter_texts = [label_to_text(s.edge) + ':' for s in samples]
    sents_0 = [s.head_text for s in samples]
    sents_1 = [inter_text + ('none' if s.tail_text == '' else s.tail_text)
               for s, inter_text in zip(samples, inter_texts)]

    sample_ids = []
    token_type_ids = []
    sample_masks = []
    for s1, s2 in zip(sents_0, sents_1):
        if isinstance(tokenizer, RobertaTokenizer):
            r = tokenizer.encode_plus(s1, s2, max_length=max_seq_length, add_prefix_space=True)
        else:
            r = tokenizer.encode_plus(s1, s2, max_length=max_seq_length)
        if 'num_truncated_tokens' in r:
            logging.warning('Sample too long, %d tokens truncated ("%s"): ' %
                         (r['num_truncated_tokens'], tokenizer.decode(r['overflowing_tokens'])) + s1 + ' | ' + s2)
        s = r['input_ids']
        pad_len = max_seq_length - len(s)
        s += [tokenizer.pad_token_id] * pad_len
        sample_ids.append(s)

        s = r['token_type_ids']
        s += [tokenizer.pad_token_id] * pad_len
        token_type_ids.append(s)

        s = [1] * (max_seq_length - pad_len) + [0] * pad_len
        sample_masks.append(s)

    sample_ids = torch.LongTensor(sample_ids)
    token_type_ids = torch.LongTensor(token_type_ids)
    sample_masks = torch.LongTensor(sample_masks)

    labels = torch.LongTensor([1 if s.type == SampleType.TRUTH else 0 for s in samples])
    return Example(input_ids=sample_ids, input_mask=sample_masks, token_type_ids=token_type_ids,
                   labels=labels)

class Graph:
    def __init__(self, base_path, hparams):
        self._hparams = hparams

        self.nodes = []
        lines = open(os.path.join(base_path, 'nodes.tsv')).read().splitlines()[1:]
        for i in tqdm.tqdm(range(len(lines))):
            l = lines[i].split('\t')
            l = [t if Node._field_types[Node._fields[i]] is str else json.loads(t) for i, t in enumerate(l)]
            l[3] = [EntityMention(*e) for e in l[3]]
            n = Node(*l)
            self.nodes.append(n)

        self.edges = []
        self._graph = defaultdict(dict)
        lines = open(os.path.join(base_path, 'edges.tsv')).read().splitlines()[1:]
        for i in tqdm.tqdm(range(len(lines))):
            l = lines[i].split('\t')
            l = [t if Edge._field_types[Edge._fields[i]] is str else json.loads(t) for i, t in enumerate(l)]
            e = Edge(*l)
            self.edges.append(e)
            self._graph[(e.head_id, e.tail_id)][e.label] = i

        self.skeletons = []
        lines = open(os.path.join(base_path, 'skeletons.tsv')).read().splitlines()
        for i in tqdm.tqdm(range(len(lines))):
            l = lines[i].split('\t')
            l = [t if Skeleton._field_types[Skeleton._fields[i]] is str else json.loads(t) for i, t in enumerate(l)]
            s = Skeleton(*l)
            self.skeletons.append(s)
        logging.info('KG built, with %d nodes, %d edges, %d skeletons' %
                     (len(self.nodes), len(self.edges), len(self.skeletons)))

    def find_edge_by_hrt(self, head_id, edge_label, tail_id):
        if (head_id, tail_id) in self._graph and edge_label in self._graph[(head_id, tail_id)]:
            return self._graph[(head_id, tail_id)][edge_label]
        else:
            return None

    def find_node_by_skeleton_and_concept(self, skeleton_id: int, concept: str):
        for node_id, entity_mention_id in self.skeletons[skeleton_id].subs:
            mention = self.nodes[node_id].concepts[entity_mention_id]
            if mention.base_concept == concept:
                return node_id, entity_mention_id
        return None


class ProposeFailError(ValueError):
    pass


class ConceptualizeProposer:
    def __init__(self, kg: Graph, hparams):
        self._concept_proposer = Proposer()
        self._kg = kg
        self._sub_mode = hparams.entity_sub_mode
        self._concept_score = hparams.concept_score
        self._weighted = hparams.score_weighted
        self._random_mode = hparams.random_select_mode
        self._n_candidates = hparams.n_candidates
        self._max_entity_word_inc = hparams.max_entity_word_inc

    def find_node_from_substitution(self, node: Node, sub: Substitution):
        entity_mention_id = sub.entity_mention_id
        skeleton_id = node.concepts[entity_mention_id].skeleton_id
        result = self._kg.find_node_by_skeleton_and_concept(skeleton_id, sub.alt_slice)
        if result is None:
            return None
        else:
            return result[0]

    def propose_substitutions(self, node: Node) -> List[Substitution]:
        if self._sub_mode == 'conceptualize':
            return self._concept_proposer.conceptualize(
                node.text, node.concepts, False, score_method=self._concept_score,
                top_k=self._n_candidates, max_entity_word_inc=self._max_entity_word_inc)
        elif self._sub_mode == 'random_entity':
            return self._concept_proposer.random_substitution(
                node.text, node.concepts, mode=self._random_mode, top_k=self._n_candidates, weighted=self._weighted,
                max_entity_word_inc=self._max_entity_word_inc)
        else:
            raise ValueError('Unsupported substitution mode: ' + self._sub_mode)

    def propose(self, e: Edge, return_all=False) -> Union[EdgeSample, List[EdgeSample]]:
        head = self._kg.nodes[e.head_id]
        tail = self._kg.nodes[e.tail_id]
        if len(head.concepts) + len(tail.concepts) == 0:
            raise ProposeFailError()

        head_subs = self.propose_substitutions(head)
        tail_subs = self.propose_substitutions(tail)

        head_sub_to_nodes = [self.find_node_from_substitution(head, sub) for sub in head_subs]
        tail_sub_to_nodes = [self.find_node_from_substitution(tail, sub) for sub in tail_subs]

        proposals = []
        head_weights = []
        tail_weights = []
        for i in range(len(head_subs)):
            if head_sub_to_nodes[i] is None or self._kg.find_edge_by_hrt(head_sub_to_nodes[i], e.label,
                                                                         e.tail_id) is None:
                if (head_subs[i].alt_text, e.label, tail.text) in exclude_text_samples:
                    continue
                proposals.append(EdgeSample(head_subs[i].alt_text, e.label, tail.text, SampleType.CC))
                head_weights.append(head_subs[i].weight)

        if len(head_weights) > 0:
            head_weights = list(np.asarray(head_weights) / sum(head_weights) / len(head_weights))

        for i in range(len(tail_subs)):
            if tail_sub_to_nodes[i] is None or self._kg.find_edge_by_hrt(e.head_id, e.label,
                                                                         tail_sub_to_nodes[i]) is None:
                if (head.text, e.label, tail_subs[i].alt_text) in exclude_text_samples:
                    continue
                proposals.append(EdgeSample(head.text, e.label, tail_subs[i].alt_text, SampleType.CC))
                tail_weights.append(tail_subs[i].weight)

        if len(tail_weights) > 0:
            tail_weights = list(np.asarray(tail_weights) / sum(tail_weights) / len(tail_weights))

        weights = head_weights + tail_weights
        if len(proposals) == 0:
            raise ProposeFailError()
        if return_all:
            return proposals
        if self._weighted:
            proposal = random.choices(proposals, weights=weights)[0]
        else:
            proposal = random.choice(proposals)
        return proposal


class EdgeComponentProposer:
    def __init__(self, kg: Graph, hparams):
        self._kg = kg
        self._ht_symmetry = hparams.ht_symmetry
        if hparams.ht_symmetry:
            self._heads_pool = self._tails_pool = list(range(len(kg.nodes)))
        else:
            heads_pool = set()
            tails_pool = defaultdict(set)

            for e in kg.edges:
                heads_pool.add(e.head_id)
                tails_pool[e.label].add(e.tail_id)
            heads_pool = list(heads_pool)
            for key in tails_pool.keys():
                tails_pool[key] = list(tails_pool[key])
            self._heads_pool = heads_pool
            self._tails_pool = tails_pool

    def propose(self, e: Edge) -> EdgeSample:
        while True:
            lot = random.random()
            if self._ht_symmetry:
                thres = 0.5
                tails_pool = self._tails_pool
            else:
                thres = len(self._heads_pool) / (len(self._heads_pool) + len(self._tails_pool[e.label]))
                tails_pool = self._tails_pool[e.label]
            if lot < thres:
                head_id = random.choice(self._heads_pool)
                tail_id = e.tail_id
            else:
                head_id = e.head_id
                tail_id = random.choice(tails_pool)
            if self._kg.find_edge_by_hrt(head_id, e.label, tail_id) is not None:
                continue
            head = self._kg.nodes[head_id]
            tail = self._kg.nodes[tail_id]
            if (head.text, e.label, tail.text) in exclude_text_samples:
                continue
            return EdgeSample(head.text, e.label, tail.text, SampleType.NODE)


class DataFeeder:
    def __init__(self, base_path, hparams, tokenizer: tokenization_bert.PreTrainedTokenizer, mode):
        logging.info('Initializing DataFeeder [%s]...' % mode)
        self._tokenizer = tokenizer
        self._hparams = hparams

        self._kg = Graph(base_path, hparams)

        self._offset = 0
        self._epoch = 0
        self._indices = list(range(len(self._kg.edges)))
        self._mode = mode
        if mode == 'train':
            self._sampling_rate = hparams.sampling_rate
            self._batch_size = hparams.train_batch_size
        elif mode == 'dev':
            self._sampling_rate = hparams.sampling_rate
            self._batch_size = hparams.eval_batch_size
        elif mode == 'test':
            self._sampling_rate = 0.0
            self._batch_size = hparams.eval_batch_size
        else:
            raise ValueError('Unsupported mode')
        self._indices = [c for c in self._indices if self._kg.edges[c].split == mode]
        if mode in ['train']:
            random.shuffle(self._indices)
        logging.info('DataFeeder [%s] collected %d samples' % (mode, len(self._indices)))

        self._proposers = []
        self._proposer_weights = []
        if hparams.conceptualize_rate > 0:
            self._proposers.append(ConceptualizeProposer(self._kg, hparams))
            self._proposer_weights.append(hparams.conceptualize_rate)
        if hparams.conceptualize_rate < 1:
            self._proposers.append(EdgeComponentProposer(self._kg, hparams))
            self._proposer_weights.append(1 - hparams.conceptualize_rate)

        logging.info('DataFeeder [%s] initialized' % mode)

    def produce_sample(self, base_sample_id, use_true_sample, false_proposer) -> EdgeSample:
        e = self._kg.edges[base_sample_id]
        if use_true_sample:
            return EdgeSample(self._kg.nodes[e.head_id].text, e.label, self._kg.nodes[e.tail_id].text, SampleType.TRUTH)
        else:
            return false_proposer.propose(e)

    def next_sample(self, use_true_sample, false_proposer, fail_retry=True) -> EdgeSample:
        if self._offset >= len(self._indices):
            self._offset = 0
            self._epoch += 1
            if self._mode in ['train']:
                random.shuffle(self._indices)
        sample_id = self._indices[self._offset]
        self._offset += 1
        e = self._kg.edges[sample_id]
        try:
            s = self.produce_sample(sample_id, use_true_sample, false_proposer)
            self._base_sample_ids.append(sample_id)
            return s
        except Exception as exc:
            if fail_retry and isinstance(exc, ProposeFailError):
                return self.next_sample(use_true_sample, false_proposer, fail_retry)  # Retry
            traceback.print_exc()
            raise exc

    def next_batch(self) -> Optional[Example]:
        while True:
            self._base_sample_ids = []
            if self._mode == 'train':
                n_samples = self._batch_size
            else:
                n_samples = min(self._batch_size, len(self._indices) - self._offset)
                if n_samples == 0:
                    self._offset = 0
                    return None
            use_true_sample = random.choices([True, False], k=n_samples, weights=
            [self._sampling_rate, 1 - self._sampling_rate])
            false_proposers = random.choices(self._proposers, k=n_samples, weights=self._proposer_weights)
            samples = [self.next_sample(use_true_sample[k], false_proposers[k]) for k in range(n_samples)]
            self._last_samples = samples
            return build_batch_from_edge(
                tokenizer=self._tokenizer, max_seq_length=self._hparams.max_seq_length,
                samples=samples)

# Format for text samples: head_text, type, tail_text, head_id | '', tail_id | '', label
class ExternalTextFeeder:
    def __init__(self, data_path, hparams, tokenizer: tokenization_bert.PreTrainedTokenizer, mode):
        logging.info('Initializing ExternalTextFeeder [%s] from %s...' % (mode, data_path))
        self._tokenizer = tokenizer
        self._hparams = hparams

        lines = open(data_path).read().splitlines()
        lines = [l.split('\t') for l in lines]
        self._samples = lines

        self._offset = 0
        self._epoch = 0
        self._indices = list(range(len(self._samples)))
        self._mode = mode
        if mode == 'train':
            self._batch_size = hparams.train_batch_size
        elif mode == 'dev':
            self._batch_size = hparams.eval_batch_size
        elif mode == 'test':
            self._batch_size = hparams.eval_batch_size
        else:
            raise ValueError('Unsupported mode')

        for line in lines:
            exclude_text_samples.add((line[0], line[1], line[2]))

        if mode == 'train':
            random.shuffle(self._indices)
        logging.info('DataFeeder [%s] collected %d samples' % (mode, len(self._indices)))

    def next_sample(self):
        if self._offset >= len(self._indices):
            self._offset = 0
            self._epoch += 1
            if self._mode == 'train':
                random.shuffle(self._indices)
        sample = self._samples[self._indices[self._offset]]
        label = int(sample[-1])
        e = EdgeSample(head_text=sample[0], edge=sample[1], tail_text=sample[2],
                       type=(SampleType.TRUTH if label == 1 else SampleType.NAT))
        self._offset += 1
        return e

    def next_batch(self) -> Optional[Example]:
        while True:
            self._base_sample_ids = []
            if self._mode == 'train':
                n_samples = self._batch_size
            else:
                n_samples = min(self._batch_size, len(self._indices) - self._offset)
                if n_samples == 0:
                    self._offset = 0
                    return None
            samples = [self.next_sample() for k in range(n_samples)]
            self._last_samples = samples
            return build_batch_from_edge(
                tokenizer=self._tokenizer, max_seq_length=self._hparams.max_seq_length,
                samples=samples)

# Format for text samples: head_text, type, tail_text, head_id | '', tail_id | '', label
class PrebuiltTrainFeeder:
    def __init__(self, data_path, hparams, tokenizer: tokenization_bert.PreTrainedTokenizer, mode):
        if mode != 'train':
            raise NotImplementedError()
        logging.info('Initializing PrebuiltTrainFeeder [%s]...' % mode)
        self._tokenizer = tokenizer
        self._hparams = hparams
        self._sampling_rate = hparams.sampling_rate
        self._conceptualize_rate = hparams.conceptualize_rate

        lines = open(data_path).read().splitlines()
        lines = [None if len(l) == 0 else l.split('\t') for l in lines]
        self._samples = [(lines[i], lines[i+1], lines[i+2]) for i in range(0, len(lines), 3)]

        self._offset = 0
        self._epoch = 0
        self._indices = list(range(len(self._samples)))
        self._batch_size = hparams.train_batch_size
        random.shuffle(self._indices)
        logging.info('DataFeeder [%s] collected %d samples' % (mode, len(self._indices)))

    def next_sample(self, choice):
        if self._offset >= len(self._indices):
            self._offset = 0
            self._epoch += 1
            random.shuffle(self._indices)
        sample = self._samples[self._indices[self._offset]][choice]
        self._offset += 1
        if sample is None: # Retry
            return self.next_sample(choice)
        label = int(sample[-1])
        e = EdgeSample(head_text=sample[0], edge=sample[1], tail_text=sample[2],
                       type=(SampleType.TRUTH if label == 1 else (SampleType.NODE if choice == 1 else SampleType.CC)))
        return e

    def next_batch(self) -> Optional[Example]:
        while True:
            n_samples = self._batch_size
            use_true_sample = random.choices([0, 1], k=n_samples, weights=
            [self._sampling_rate, 1 - self._sampling_rate])
            false_proposer = random.choices([1, 2], k=n_samples, weights=
            [1 - self._conceptualize_rate, self._conceptualize_rate])
            samples = [self.next_sample(0 if use_true_sample[k] == 0 else false_proposer[k]) for k in range(n_samples)]
            return build_batch_from_edge(
                tokenizer=self._tokenizer, max_seq_length=self._hparams.max_seq_length,
                samples=samples)



class FullGenerationFeeder:
    def __init__(self, base_path, hparams, tokenizer: tokenization_bert.PreTrainedTokenizer, accept_splits):
        logging.info('Initializing FullGenerationFeeder')
        self._tokenizer = tokenizer
        self._hparams = hparams
        if isinstance(accept_splits, str):
            accept_splits = [accept_splits]
        self._splits = accept_splits

        self._kg = Graph(base_path, hparams)

        self._offset = 0
        self._epoch = 0
        self._indices = list(range(len(self._kg.edges)))
        self._indices = [c for c in self._indices if self._kg.edges[c].split in accept_splits]
        self._batch_size = hparams.eval_batch_size

        self._proposers = ConceptualizeProposer(self._kg, hparams)
        self._proposed_samples = [list() for _ in range(len(self._kg.edges))]

        logging.info('FullGenerationFeeder initialized')

    def next_sample(self):
        if self._offset >= len(self._indices):
            self._offset = 0
            self._epoch += 1
        sample_id = self._indices[self._offset]
        self._offset += 1
        e = self._kg.edges[sample_id]
        try:
            cands = self._proposers.propose(e, return_all=True)
            return sample_id, cands
        except:
            pass
        return None, []

    def next_batch(self) -> Optional[Example]:
        if not hasattr(self, '_rem_cands'):
            base_sample_id, cands = None, []
        else:
            base_sample_id, cands = self._rem_cands
        if self._offset == len(self._indices):
            return None
        samples = []
        base_sample_ids = []
        while len(samples) < self._batch_size and self._offset < len(self._indices):
            while len(cands) == 0 and self._offset < len(self._indices):
                base_sample_id, cands = self.next_sample()
            n_sample = cands[-1]
            cands = cands[:-1]
            self._proposed_samples[base_sample_id].append(n_sample)
            base_sample_ids.append(base_sample_id)
            samples.append(n_sample)

        self._last_samples = samples
        self._base_sample_ids = base_sample_ids
        self._rem_cands = (base_sample_id, cands)
        return build_batch_from_edge(
            tokenizer=self._tokenizer, max_seq_length=self._hparams.max_seq_length,
            samples=samples)