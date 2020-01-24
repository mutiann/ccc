import pandas as pd
import json
import tqdm
from utils import Node, Edge
from atomic_util import normalize, identify_concepts
import os, shutil
import spacy, pickle, random
from collections import defaultdict

nlp = spacy.load("en_core_web_lg")
docs = []

def build_graph():
    # Schema:
    # Node: id, text, words, concepts, event_ids, annotation_ids
    # Edge: x_id, y_id, type, inter_text, count

    base_path = r'data/atomic/tmp/'
    in_path = r"data/atomic/atomic_data/v4_atomic_all_agg.csv"
    out_node = r'nodes_s1.tsv'
    out_edges = r'edges.tsv'
    os.makedirs(base_path, exist_ok=True)

    df = pd.read_csv(in_path)
    out_node = open(os.path.join(base_path, out_node), 'w')
    out_edges = open(os.path.join(base_path, out_edges), 'w')
    df.iloc[:, 1:10] = df.iloc[:, 1:10].apply(lambda col: col.apply(json.loads))

    out_node.write('\t'.join(Node._fields) + '\n')
    out_edges.write('\t'.join(Edge._fields) + '\n')

    nodes_dict = dict()
    edges_dict = dict()

    nodes = []
    edges = []



    def build_node(text, event_id=None, annotation_id=None):
        text = normalize(text)
        if text not in nodes_dict:
            id = len(nodes)
            nodes_dict[text] = id
            event_ids = []
            annotation_ids = []
            node = Node(id, text, [], [], event_ids, annotation_ids)
            nodes.append(node)
        else:
            id = nodes_dict[text]
        if event_id is not None:
            nodes[id].event_ids.append(event_id)
        if annotation_id is not None:
            nodes[id].annotation_ids.append(annotation_id)
        return id


    def build_edge(head, tail, type):
        if (head, tail, type) not in edges_dict:
            id = len(edges)
            edges_dict[(head, tail, type)] = id
            inter_text = ''
            edge = Edge(head, tail, type, inter_text, 1, '')
            edges.append(edge)
        else:
            id = edges_dict[(head, tail, type)]
            edges[id] = edges[id]._replace(count=edges[id].count + 1)
        return id


    build_node('none')

    n_annotation = 0

    for idx, line in tqdm.tqdm(df.iterrows()):
        head = build_node(line[0], event_id=idx)

        for i in range(1, 10):
            item = line[i]
            tails = []
            for t in item:
                tail = build_node(t, annotation_id=n_annotation)
                n_annotation += 1
                build_edge(head, tail, df.columns[i])

    for e in edges:
        out_edges.write('\t'.join([json.dumps(t) if t is list else str(t) for t in e]) + '\n')

    for n in tqdm.tqdm(nodes):
        doc = nlp(n.text)
        docs.append(doc)
    pickle.dump(docs, open(os.path.join(base_path, 'parsed_nodes'), 'wb'))

    for n in nodes:
        out_node.write('\t'.join(list([json.dumps(t) if t is list else str(t) for t in n])) + '\n')

def collect_concepts():
    # Schema:
    # Node: id, text, words, concepts, event_ids, annotation_ids, tokens
    base_path = r'data/atomic/tmp'
    in_path = r'nodes_s1.tsv'
    out_path = r'nodes_s2.tsv'
    concept_path = r'data/probase/concepts'
    concept_set = pickle.load(open(concept_path, 'rb'))
    docs = pickle.load(open(os.path.join(base_path, 'parsed_nodes'), 'rb'))
    fw = open(os.path.join(base_path, out_path), 'w')

    lines = open(os.path.join(base_path, in_path)).read().splitlines()
    head, lines = lines[0], lines[1:]
    print('Doc sample: ')
    print(lines[-10])
    for tok in docs[-10]:
        print(tok.text, tok.dep_, tok.pos_, tok.head.text)

    print('Start checking...')
    results = []

    for i, line in tqdm.tqdm(enumerate(lines)):
        line = line.split('\t')
        sent = line[1]
        logs = []
        doc = docs[i]
        words = [[t.idx, t.idx + len(t)] for t in doc]
        line[2] = json.dumps(words)
        concepts = identify_concepts(sent, doc, logs, concept_set)
        line[3] = json.dumps(concepts)
        line = '\t'.join(line)
        results.append(line + '\n')

    fw.write(head + '\n')

    fw.writelines(results)

def collect_skeletons():
    # Schema:
    # Node: id, text, words, concepts, event_ids, annotation_ids
    # Edge: x_id, y_id, type, inter_text, count
    # Dev: x_id, y_id

    base_path = r'data/atomic/tmp'
    in_path = r'nodes_s2.tsv'
    out_dev = r'skeletons.tsv'

    lines = open(os.path.join(base_path, in_path)).read().splitlines()
    line_head, lines = lines[0], lines[1:]
    docs = pickle.load(open(os.path.join(base_path, 'parsed_nodes'), 'rb'))

    out_dev = open(os.path.join(base_path, out_dev), 'w')

    skeletons = defaultdict(list)
    skeleton_ids = {}

    print('Doc sample: ')
    print(lines[-10])
    for tok in docs[-10]:
        print(tok.text, tok.dep_, tok.pos_, tok.tag_, tok.head.text)

    u_lines = [line_head + '\n']

    for i, item in tqdm.tqdm(enumerate(lines)):
        item = item.split('\t')
        concepts = json.loads(item[3])
        text = item[1]
        mentions = []
        for j, (l_i, l, r_i, r, det_idx, det_text, cent_i, cent_idx, base_concept) in enumerate(concepts):
            if len(det_text) == 0:
                skeleton_text = text[:l] + '[SEP]' + text[r:]
            else:
                skeleton_text = text[:det_idx] + text[det_idx + len(det_text) + 1:l] + '[SEP]' + text[r:]
            if skeleton_text.startswith('to '):
                skeleton_text = skeleton_text[3:]
                first_ess = 1
            elif skeleton_text[:5] in ['Alex ', 'they '] and docs[i][1].pos_ in ['VERB', 'AUX']:
                skeleton_text = skeleton_text[5:]
                first_ess = 1
            elif skeleton_text[:4] in ['Bob ', 'she '] and docs[i][1].pos_ in ['VERB', 'AUX']:
                skeleton_text = skeleton_text[4:]
                first_ess = 1
            elif skeleton_text[:3] in ['he ', ] and docs[i][1].pos_ in ['VERB', 'AUX']:
                skeleton_text = skeleton_text[3:]
                first_ess = 1
            else:
                first_ess = 0
            if skeleton_text.startswith(docs[i][first_ess].text) and docs[i][first_ess].tag_ in ['VBZ', 'VBD', 'VBG',
                                                                                                 'NNS']:
                skeleton_text = docs[i][first_ess].lemma_ + skeleton_text[len(docs[i][first_ess]):]
            skeletons[skeleton_text].append([i, j])
            if skeleton_text not in skeleton_ids:
                skeleton_ids[skeleton_text] = len(skeleton_ids)
            mentions.append([l_i, l, r_i, r, det_idx, det_text, cent_i, cent_idx,
                             base_concept, skeleton_ids[skeleton_text]])
        item[3] = json.dumps(mentions)
        u_lines.append('\t'.join(item) + '\n')

    items = list(skeletons.items())
    print('Total %d skeletons, e.g. %s' % (len(skeletons), items[10][0]))
    for text, branches in items:
        out_dev.write('%s\t%s\n' % (text, json.dumps(branches)))

    fw = open(os.path.join(base_path, 'nodes.tsv'), 'w')
    fw.writelines(u_lines)


def split_o_hr():
    from datafeeder import Graph, EdgeComponentProposer, ConceptualizeProposer, ProposeFailError
    from hparams import hparams
    in_path = r'data/atomic/tmp'
    agg_path = r"data/atomic/atomic_data/v4_atomic_all_agg.csv"
    base_path = r'data/atomic'
    df = pd.read_csv(agg_path)
    train_ev = set()
    dev_ev = set()
    for i, s in enumerate(df['split']):
        if s == 'trn':
            train_ev.add(i)
        elif s == 'dev':
            dev_ev.add(i)

    G = Graph(in_path, hparams)

    train_head = set()
    for i in range(len(G.nodes)):
        for h in G.nodes[i].event_ids:
            if h in train_ev:
                train_head.add(i)
                break
        if len(G.nodes[i].event_ids) > 1:
            print(G.nodes[i])

    print('Collected %d train heads' % len(train_head))

    dev_head = set()
    for i in range(len(G.nodes)):
        for h in G.nodes[i].event_ids:
            if h in dev_ev:
                dev_head.add(i)
                break
        if len(G.nodes[i].event_ids) > 1:
            print(G.nodes[i])

    print('Collected %d dev heads' % len(dev_head))

    splits = [None] * len(G.edges)
    n_tr = n_dev = n_test = 0
    for i, e in enumerate(G.edges):
        if e.head_id in train_head:
            splits[i] = 'train'
            n_tr += 1
        elif e.head_id in dev_head:
            splits[i] = 'dev'
            n_dev += 1
        else:
            splits[i] = 'test'
            n_test += 1

    print('%d trains %d dev %d test' % (n_tr, n_dev, n_test))

    cc_dev_cands = []
    ec_dev_cands = []
    cc_tst_cands = []
    ec_tst_cands = []

    hparams.conceptualize_rate = 1.0
    EP = EdgeComponentProposer(G, hparams)
    CP = ConceptualizeProposer(G, hparams)

    for i in tqdm.tqdm(range(len(splits))):
        if splits[i] == 'train':
            continue
        if splits[i] == 'dev':
            ec_cands = ec_dev_cands
            cc_cands = cc_dev_cands
        else:
            ec_cands = ec_tst_cands
            cc_cands = cc_tst_cands
        p = EP.propose(G.edges[i])
        ec_cands.append([i, p])
        try:
            p = CP.propose(G.edges[i])
            cc_cands.append([i, p])
        except ProposeFailError as e:
            continue
    print('EC devs: %d, tests: %d; NS devs: %d, tests: %d' %
          (len(cc_dev_cands), len(cc_tst_cands), len(ec_dev_cands), len(ec_tst_cands)))

    files = ['nodes.tsv', 'skeletons.tsv']
    for f in files:
        shutil.copy(os.path.join(in_path, f), os.path.join(base_path))

    cc_dev_cands.sort(key=lambda x: x[0])
    fw = open(os.path.join(base_path, 'concept_contrastive_dev.tsv'), 'w')
    for i, p in cc_dev_cands:
        e = G.edges[i]
        fw.write('\t'.join([G.nodes[e.head_id].text, e.label, G.nodes[e.tail_id].text,
                            str(e.head_id), str(e.tail_id), '1']) + '\n')
        fw.write('\t'.join([p[0], p[1], p[2], '', '', '0']) + '\n')

    cc_tst_cands.sort(key=lambda x: x[0])
    fw = open(os.path.join(base_path, 'concept_contrastive_test.tsv'), 'w')
    for i, p in cc_tst_cands:
        e = G.edges[i]
        fw.write('\t'.join([G.nodes[e.head_id].text, e.label, G.nodes[e.tail_id].text,
                            str(e.head_id), str(e.tail_id), '1']) + '\n')
        fw.write('\t'.join([p[0], p[1], p[2], '', '', '0']) + '\n')
    print('Finished concept_contrastive')

    ec_dev_cands.sort(key=lambda x: x[0])
    fw = open(os.path.join(base_path, 'node_contrastive_dev.tsv'), 'w')
    for i, p in ec_dev_cands:
        e = G.edges[i]
        fw.write('\t'.join([G.nodes[e.head_id].text, e.label, G.nodes[e.tail_id].text,
                            str(e.head_id), str(e.tail_id), '1']) + '\n')
        fw.write('\t'.join([p[0], p[1], p[2], '', '', '0']) + '\n')

    ec_tst_cands.sort(key=lambda x: x[0])
    fw = open(os.path.join(base_path, 'node_contrastive_test.tsv'), 'w')
    for i, p in ec_tst_cands:
        e = G.edges[i]
        fw.write('\t'.join([G.nodes[e.head_id].text, e.label, G.nodes[e.tail_id].text,
                            str(e.head_id), str(e.tail_id), '1']) + '\n')
        fw.write('\t'.join([p[0], p[1], p[2], '', '', '0']) + '\n')
    print('Finished node_contrastive')

    lines = open(os.path.join(in_path, 'edges.tsv')).read().splitlines()
    fw = open(os.path.join(base_path, 'edges.tsv'), 'w')
    edge_head, lines = lines[0], lines[1:]
    fw.write(edge_head + '\n')

    for i, l in enumerate(lines):
        l = l.split('\t')
        l[-1] = splits[i]
        l = '\t'.join(l)
        fw.write(l + '\n')

    print('Finished edges')

if __name__ == '__main__':
    build_graph()
    collect_concepts()
    collect_skeletons()
    split_o_hr()
