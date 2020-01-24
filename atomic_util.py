import inflect
import re
import unidecode
_inflect = inflect.engine()

def normalize(text: str):
    text = text.lower()
    text = unidecode.unidecode_expect_ascii(text)
    if text == 'none':
        return ''
    text = text.replace('___', '_')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(a )+', 'a ', text)
    text = re.sub(r'\b(an )+', 'an ', text)
    text = re.sub(r'\b(the )+', 'the ', text)
    text = re.sub(r'\b(person( |)|)x(?!-)\b', 'Alex', text)
    text = re.sub(r'\b(person( |)|)y(?!-)\b', 'Bob', text)
    text = re.sub(r'\b(person( |)|)z(?!-)\b', 'Charlie', text)
    if not text.endswith('.'):
        text += '.'
    return text

def collect_words(text):
    indices = []
    for m in re.finditer(r'\w+', text):
        index, item = m.start(), m.group()
        indices.append([index, index + len(item)])
    return indices

def identify_concepts(sent, doc, logs=None, concept_set=None):
    if logs is None:
        logs = []

    np_labels = ['nsubj', 'nsubjpass', 'dobj', 'pobj']
    extra_np_labels = ['iobj', 'poss', 'dative', 'npadvmod', 'appos', 'oprd', 'attr',
                       'compound', 'nmod', 'conj', 'ccomp', 'acomp', 'xcomp', 'relcl', 'ROOT']
    np_labels += extra_np_labels
    pos_labels = ['NOUN', 'PROPN']

    def part_of_subtok(tok):
        anc = tok.ancestors
        return any([a.dep_ == 'subtok' for a in anc])

    entities = [tok for tok in doc if (tok.dep_ in np_labels and tok.pos_ in pos_labels)]
    concepts = []
    for tok in entities:
        if part_of_subtok(tok):
            continue
        if tok.text in ['Alex', 'Bob', 'Charlie']:
            continue
        sign_to_ban = ['prt', 'aux', 'auxpass']
        if any([t.dep_ in sign_to_ban for t in tok.children]):
            continue
        left_child = list(tok.lefts) + [tok]
        right_child = [tok] + list(tok.rights)

        noun_phrase_spec = ['det', 'amod', 'prep', 'compound', 'nmod', 'punct', 'poss', 'advmod',
                            'cc', 'conj', 'dep', 'agent', 'intj']
        noun_phrase_seps = ['relcl', 'acl', 'advcl', 'nsubj', 'dobj', 'pobj', 'iobj', 'csubj', 'nsubjpass',
                            'ccomp', 'appos', 'xcomp', 'acomp', 'case', 'nummod', 'predet', 'oprd', 'npadvmod',
                            'neg', 'mark', 'preconj', 'dative', 'expl']
        seps = [t for t in tok.children if t.dep_ in noun_phrase_seps]
        for s in seps:
            if s.i < tok.i:
                left_child = [l for l in left_child if l.i > s.right_edge.i]
            if s.i > tok.i:
                right_child = [l for l in right_child if l.i < s.left_edge.i]
        # if len(seps) > 0:
        #     print('Original: ', sent, '|', tok, ' -> ', list(tok.children))
        #     print(list(tok.subtree))
        #     print('Seps: ', seps)
        #     print('Cleaned: ', left_child, right_child)

        det = [t for t in tok.lefts if t.dep_ == 'det' and t.text in ['a', 'an', 'the'] and
               (t.i + 1 < len(doc) and doc.text[doc[t.i + 1].idx - 1] == ' ')]
        if len(det) == 1:
            det = det[0]
            left_child = [l for l in left_child if l.idx > det.idx]
        elif len(det) > 1:
            print(sent, '|', det, '|', tok.text)
            det = det[0]
        else:
            det = None
        for l in left_child:
            for r in right_child:
                text = sent[l.idx: tok.idx] + tok.lemma_ + sent[tok.idx + len(tok): r.idx + len(r)]
                if concept_set is None or (text in concept_set):
                    if det is None:
                        det_idx = l.idx
                        det_text = ''
                    else:
                        det_idx = det.idx
                        det_text = det.text
                    concepts.append([l.i, l.idx, r.i, r.idx + len(r), det_idx, det_text, tok.i, tok.idx, text])
                    logs.append(det_text + ' | ' + text + '\n')
    return concepts