from typing import NamedTuple, List, Tuple
import logging, os, sys
from hparams import hparams

# l_i, l, r_i, r, det_idx, det_text, cent_i, cent_idx, base_concept, skeleton_id
class EntityMention(NamedTuple):
    l_i: int
    l: int
    r_i: int
    r: int
    det_idx: int
    det_text: str
    cent_i: int
    cent_idx: int
    base_concept: str
    skeleton_id: int


class Node(NamedTuple):
    id: int
    text: str
    words: list
    concepts: List[EntityMention]
    # Only meaningful in Atomic
    event_ids: list
    annotation_ids: list


class Edge(NamedTuple):
    head_id: int
    tail_id: int
    label: str
    inter_text: str
    count: int
    split: str


class Skeleton(NamedTuple):
    text: str
    subs: list # [(node_id, entity_mention_id)]

def label_to_text_atomic(label: str) -> str:
    if label[0] == 'x':
        t = 'alex'
    elif label[0] == 'o':
        t = 'others'
    else:
        raise ValueError('Invalid label: ' + label)
    if label[1:] in ['Effect', 'React', 'Need', 'Want']:
        t += ' ' + label[1:].lower()
    elif label[1:] == 'Attr':
        assert t == 'alex', 'Invalid label: ' + label
        t += ' attribute'
    elif label[1:] == 'Intent':
        assert t == 'alex', 'Invalid label: ' + label
        t += ' intent'
    else:
        raise ValueError('Invalid label: ' + label)
    if label == 'xNeed':
        t = 'before, ' + t
    elif label == 'xIntent':
        t = 'because, ' + t
    else:
        t = 'therefore, ' + t
    return t

def label_to_text_aser(label: str) -> str:
    types = ['Precedence', 'Succession', 'Synchronous', 'Reason', 'Result', 'Condition', 'Contrast', 'Concession',
             'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception',
             'Co_Occurrence']
    texts = ['then', 'after', 'meanwhile', 'because', 'therefore', 'if', 'but', 'although', 'and', 'for instance',
             'in other words', 'or', 'instead', 'except', '']
    mapping = dict(zip(types, texts))
    return mapping[label]

def label_to_text(label: str) -> str:
    if hparams.expr == 'atomic':
        return label_to_text_atomic(label)
    elif hparams.expr == 'aser':
        return label_to_text_aser(label)
    else:
        raise ValueError()

class ValueWindow:
  def __init__(self, window_size=100):
    self._window_size = window_size
    self._values = []

  def append(self, x):
    self._values = self._values[-(self._window_size - 1):] + [x]

  @property
  def sum(self):
    return sum(self._values)

  @property
  def count(self):
    return len(self._values)

  @property
  def average(self):
    return self.sum / max(1, self.count)

  def reset(self):
    self._values = []

def set_logger(output_path=None):
    fmt = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handlers = []
    if output_path is not None:
        h = logging.FileHandler(output_path)
        h.setFormatter(fmt)
        handlers.append(h)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    handlers.append(h)
    logging.basicConfig(handlers=handlers, level=logging.INFO)