import pickle
import sqlite3
from collections import namedtuple
import random, os, shutil, json
import pandas as pd
import tqdm
import unidecode
from utils import Node, Edge, Skeleton, EntityMention
from datafeeder import ConceptualizeProposer, EdgeComponentProposer, Graph, ProposeFailError
from collections import defaultdict
import copy, re
from hparams import hparams
from transformers import BertTokenizer
import numpy as np

tk = BertTokenizer.from_pretrained('bert-base-uncased')

types = ['Precedence', 'Succession', 'Synchronous', 'Reason', 'Result', 'Condition', 'Contrast', 'Concession',
         'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception',
         'Co_Occurrence']

def normalize(text):
    text = re.sub(r'((http(\S)*)|(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}))',
                  '[URL]', text)
    text = re.sub(r'[0-9][0-9\-\.,]{3,}[0-9]', r'#', text)
    return text

def KG_retrieve(in_path, output_path):
    conn = sqlite3.connect(in_path)
    os.makedirs(output_path, exist_ok=True)
    cur = conn.cursor()

    node_present = {}

    cur.execute('select * from Relations')
    edge_samples = []
    for i, e in tqdm.tqdm(enumerate(cur)):
        l = list(e)[1:]
        counts = l[2:-1] # CoOccurrence ignored
        if all([int(c) == 0 for c in counts]):
            continue
        if l[0] not in node_present:
            node_present[l[0]] = len(node_present)
        l[0] = node_present[l[0]]
        if l[1] not in node_present:
            node_present[l[1]] = len(node_present)
        l[1] = node_present[l[1]]
        for j in range(len(counts)):
            c = int(counts[j])
            if c > 0:
                edge_samples.append([l[0], l[1], j, c]) # nodeid_v1

    print('%d edge collected' % len(edge_samples))

    cur.execute('select * from Eventualities')
    node_samples = []
    # _id verbs skeleton_words_clean skeleton_words words pattern frequency
    lengths = defaultdict(int)
    for i, e in tqdm.tqdm(enumerate(cur)):
        l = list(e)
        if l[0] not in node_present:
            continue
        l[0] = node_present[l[0]] # nodeid_v1
        l[-1] = str(int(l[-1]))
        for j in range(1, 5):
            l[j] = unidecode.unidecode(l[j])
        l[2] = normalize(l[2])
        l[3] = normalize(l[3])
        l[4] = normalize(l[4])
        tlen = len(tk.tokenize(l[4]))
        if tlen > 20:
            print(tlen, l[4])
            continue
        lengths[tlen] += 1
        node_samples.append(l)
    node_samples.sort()

    node_mapping = {}
    for i, n in enumerate(node_samples):
        node_mapping[n[0]] = i # nodeid_v1 -> nodeid_v2
        n[0] = i

    print('%d nodes' % len(node_samples))
    lengths = list(lengths.items())
    lengths.sort()
    for k, v in lengths:
        print(k, v)

    re_edge_samples = []
    for e in edge_samples:
        if e[0] not in node_mapping or e[1] not in node_mapping:
            continue
        e[0] = node_mapping[e[0]]
        e[1] = node_mapping[e[1]]
        re_edge_samples.append(e)
    edge_samples = re_edge_samples
    print('Rescanned: %d edges' % len(edge_samples))

    fw = open(os.path.join(output_path, 'edges.tsv'), 'w')
    fw.write('\t'.join(['head_id', 'tail_id', 'label', 'count']) + '\n')
    edge_samples.sort()
    for head_id, tail_id, label, count in edge_samples:
        fw.write('\t'.join([str(head_id), str(tail_id), types[label], str(count)]) + '\n')
    fw.close()

    fw = open(os.path.join(output_path, 'nodes.tsv'), 'w')
    fw.write('\t'.join(['verbs', 'skeleton_words_clean', 'skeleton_words', 'words', 'pattern', 'count']) + '\n')
    for l in node_samples:
        l = l[1:]
        l[-1] = str(l[-1])
        fw.write('\t'.join(l) + '\n')
    fw.close()


def build_graph(base_path, out_path, concept_path):
    print('Loading nodes...')
    lines = open(os.path.join(base_path, 'nodes.tsv')).read().splitlines()
    node_head, nodes = lines[0], lines[1:]
    node_head = node_head.split('\t')
    for i in tqdm.tqdm(range(len(nodes))):
        n = nodes[i].split('\t')
        n[1] = n[1].split(' ')
        n[2] = n[2].split(' ')
        n[3] = n[3].split(' ')
        n[4] = n[4].split('-')
        n[5] = int(n[5])
        nodes[i] = n

    print('Loading edges...')
    lines = open(os.path.join(base_path, 'edges.tsv')).read().splitlines()
    lines = [l.split('\t') for l in lines]
    edge_head, edges = lines[0], lines[1:]
    for e in edges:
        e[0] = int(e[0])
        e[1] = int(e[1])
        e[-1] = int(e[-1])

    # What we have:
    # 'verbs', 'skeleton_words_clean', 'skeleton_words', 'words', 'pattern', 'count'
    # 'head_id', 'tail_id', 'label', 'count'
    # What we need:
    # concepts: List[EntityMention]
    # EntityMention: l_i, l, r_i, r, det_idx, det_text, cent_i, cent_idx, base_concept, skeleton_id
    # Skeleton: text, [(node_id, entity_mention_id)]
    # Node.words/event_ids/annotation_ids & Edge.intertext are unnecessary; Edge.split would be handled later
    sk_text_to_id = dict()
    skeletons = []

    concept_set = pickle.load(open(concept_path, 'rb'))

    def get_skeleton(text: str) -> int:
        if text in sk_text_to_id:
            return sk_text_to_id[text]
        sk_text_to_id[text] = len(skeletons)
        skeletons.append((text, []))
        return len(skeletons) - 1

    def list_index(l, item, from_idx):
        for i in range(from_idx + 1, len(l)):
            if l[i] == item:
                return i
        return None

    def find_entities_normal(skeleton_words, words: list, pattern):
        last_i = -1
        results = []
        for k, (sk, p) in enumerate(zip(skeleton_words, pattern)):
            next = list_index(words, sk, last_i)
            if next is None:
                if p == 'X':
                    next = last_i
                else:
                    return None
            last_i = next
            results.append(last_i)
        return results

    dets = ['a', 'an', 'the', 'this', 'that', 'some']

    def find_entities(skeleton_words, words: list, pattern: list):
        results = find_entities_normal(skeleton_words, words, pattern)
        if results is not None or (results is None and 'X' not in pattern):
            return results
        # X-o-s-v-...
        a_pattern = pattern[-2:] + pattern[:-2]
        results = find_entities_normal(skeleton_words, words, a_pattern)
        if results is not None:
            if results[0] == -1 and words[0] != skeleton_words[0]:
                words.insert(0, skeleton_words[0])
                results = find_entities_normal(skeleton_words, words, a_pattern)
                pattern.clear()
                pattern.extend(a_pattern)
                return results
            else:
                return None
        # s-X-o-...
        a_pattern = pattern[:1] + pattern[-2:] + pattern[1:-2]
        results = find_entities_normal(skeleton_words, words, a_pattern)
        if results is not None:
            if results[1] == results[0] and words[results[0] + 1] != skeleton_words[1]:
                words.insert(results[0] + 1, skeleton_words[1])
                results = find_entities_normal(skeleton_words, words, a_pattern)
                # if not (words[results[0] + 2] != skeleton_words[2] or (results[0] + 3 < len(words) and
                #         words[results[0] + 2] in dets and words[results[0] + 3] == skeleton_words[2])):
                #     print(3.5)
                #     return None
                pattern.clear()
                pattern.extend(a_pattern)
                return results
            else:
                return None
        # s-v-X-o-...
        a_pattern = pattern[:2] + pattern[-2:] + pattern[2:-2]
        results = find_entities_normal(skeleton_words, words, a_pattern)
        if results is not None:
            if results[2] == results[1] and words[results[1] + 1] != skeleton_words[2]:
                words.insert(results[1] + 1, skeleton_words[2])
                results = find_entities_normal(skeleton_words, words, a_pattern)
                # if not (words[results[1] + 2] == skeleton_words[3] or (results[1] + 3 < len(words) and
                #         words[results[1] + 2] in dets and words[results[1] + 3] == skeleton_words[3])):
                #     print(4.5)
                #     return None
                pattern.clear()
                pattern.extend(a_pattern)
                return results
            else:
                return None
        return results

    def build_entity_mention(sk_positions, l, r, k, words, skeleton_id, offsets, base_concept):
        # sk_positions: k -> i/l/r (word index); l/r: pos in words; offsets: i -> idx (char offset)
        if k == 0:
            low = -1
        else:
            low = sk_positions[k - 1]
        det_i = l
        for i in range(l - 1, low, -1):
            if words[i] in ['a', 'the', 'an']:
                det_i = i
                break
        if det_i == l:
            det_text = ''
        else:
            det_text = words[det_i]
        det_idx = offsets[det_i]
        return EntityMention(l_i=l, l=offsets[l], r_i=r, r=offsets[r] + len(words[r]), det_idx=det_idx,
                             det_text=det_text, cent_i=r, cent_idx=offsets[r], base_concept=base_concept,
                             skeleton_id=skeleton_id)

    def build_mentions_skeletons(node_id, skeleton_words_clean, skeleton_words, words, pattern, sk_positions):
        concepts = []
        offsets = [0]
        for i in range(1, len(words)):
            offsets.append(offsets[i - 1] + len(words[i - 1]) + 1)
        text = ' '.join(words)
        last = -1
        for k, i in enumerate(sk_positions):
            if (pattern[k] not in ['s', 'o', 'spass', 'a']) or (skeleton_words[k] not in skeleton_words_clean):
                last = i
                continue
            l = r = i
            while l > last:
                if words[l] in dets:
                    break
                cand = text[offsets[l]: offsets[r] + len(words[r])]
                if cand not in concept_set:
                    l -= 1
                    continue
                alt_text = text[:offsets[l]] + '[SEP]' + text[offsets[r] + len(words[r]):]
                skeleton_id = get_skeleton(alt_text)
                skeletons[skeleton_id][1].append((node_id, len(concepts)))
                concepts.append(build_entity_mention(sk_positions, l, r, k, words, skeleton_id, offsets, cand))
                l -= 1
            last = i
        return concepts

    fw = open(os.path.join(out_path, 'nodes.tsv'), 'w')
    fw.write('id	text	words	concepts	event_ids	annotation_ids\n')

    re_node_ids = []
    n_node = 0
    # 'verbs', 'skeleton_words_clean', 'skeleton_words', 'words', 'pattern', 'count'
    print('Start building...')
    for i, n in enumerate(tqdm.tqdm(nodes)):
        try:
            if len(n[2]) != len(n[4]):
                raise ValueError()
            sk_positions = find_entities(n[2], n[3], n[4])
            if sk_positions is None:
                raise ValueError()
        except:
            re_node_ids.append(None)
            print(i, ': ', n)
            continue
        concepts = build_mentions_skeletons(n_node, n[1], n[2], n[3], n[4], sk_positions)
        fw.write('\t'.join([str(n_node), ' '.join(n[3]), '[]', json.dumps(concepts), '[]', '[]']) + '\n')
        re_node_ids.append(n_node)
        n_node += 1
    print('Total %d nodes, %d disposed' % (len(re_node_ids), len(re_node_ids) - n_node))
    assert len(re_node_ids) == len(nodes), '%d %d' % (len(re_node_ids), len(nodes))

    fw = open(os.path.join(out_path, 'edges.tsv'), 'w')
    fw.write('head_id	tail_id	label	inter_text	count	split\n')
    n_fail_edges = 0
    for i, e in enumerate(edges):
        if re_node_ids[e[0]] is None or re_node_ids[e[1]] is None:
            print(i, ': ', '[%d]' % e[0], nodes[e[0]], e[2], '[%d]' % e[1], nodes[e[1]])
            n_fail_edges += 1
            continue
        e[0] = re_node_ids[e[0]]
        e[1] = re_node_ids[e[1]]
        fw.write('\t'.join([str(e[0]), str(e[1]), e[2], '', str(e[3]), '']) + '\n')
    print('Total %d edges disposed' % n_fail_edges)

    fw = open(os.path.join(out_path, 'skeletons.tsv'), 'w')
    for text, sub in skeletons:
        fw.write('%s\t%s\n' % (text, json.dumps(sub)))

def split(in_path, base_path):
    files = ['nodes.tsv', 'edges.tsv', 'skeletons.tsv']
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for f in files:
        shutil.copy(os.path.join(in_path, f), os.path.join(base_path))

    G = Graph(in_path, hparams)
    splits = ['train'] * len(G.edges)
    n_dev = int(0.2 * len(splits))
    # n_dev = 1000
    cc_dev_cands = []
    ec_dev_cands = []

    hparams.expr = 'aser'
    hparams.ht_symmetry = True
    EP = EdgeComponentProposer(G, hparams)
    CP = ConceptualizeProposer(G, hparams)

    all_idx = list(range(len(splits)))
    idx = np.random.choice(all_idx, [n_dev], replace=False)

    print('Start producing samples..')
    for i in tqdm.tqdm(idx):
        splits[i] = 'dev'
        p = EP.propose(G.edges[i])
        ec_dev_cands.append([i, p])
        try:
            p = CP.propose(G.edges[i])
            splits[i] = 'test'
            cc_dev_cands.append([i, p])
        except ProposeFailError as e:
            continue

    total_dev = len([c for c in splits if c == 'dev'])
    print('Total dev %d, test %d, n_dev=%d' % (total_dev, len(cc_dev_cands), n_dev))

    ids = [i for i, s in enumerate(splits) if s in ['dev', 'test']]
    random.shuffle(ids)
    for i, id in enumerate(ids):
        splits[id] = 'dev' if i % 2 == 0 else 'test'

    cc_dev_cands.sort(key=lambda x: x[0])
    fw = open(os.path.join(base_path, 'concept_contrastive_dev.tsv'), 'w')
    for i, p in cc_dev_cands:
        if splits[i] != 'dev':
            continue
        e = G.edges[i]
        fw.write('\t'.join([G.nodes[e.head_id].text, e.label, G.nodes[e.tail_id].text,
                            str(e.head_id), str(e.tail_id), '1']) + '\n')
        fw.write('\t'.join([p[0], p[1], p[2], '', '', '0']) + '\n')
    fw = open(os.path.join(base_path, 'concept_contrastive_test.tsv'), 'w')
    for i, p in cc_dev_cands:
        if splits[i] != 'test':
            continue
        e = G.edges[i]
        fw.write('\t'.join([G.nodes[e.head_id].text, e.label, G.nodes[e.tail_id].text,
                            str(e.head_id), str(e.tail_id), '1']) + '\n')
        fw.write('\t'.join([p[0], p[1], p[2], '', '', '0']) + '\n')
    print('Finished concept_contrastive')

    ec_dev_cands.sort(key=lambda x: x[0])
    fw = open(os.path.join(base_path, 'node_contrastive_dev.tsv'), 'w')
    for i, p in ec_dev_cands:
        if splits[i] != 'dev':
            continue
        e = G.edges[i]
        fw.write('\t'.join([G.nodes[e.head_id].text, e.label, G.nodes[e.tail_id].text,
                            str(e.head_id), str(e.tail_id), '1']) + '\n')
        fw.write('\t'.join([p[0], p[1], p[2], '', '', '0']) + '\n')
    fw = open(os.path.join(base_path, 'node_contrastive_test.tsv'), 'w')
    for i, p in ec_dev_cands:
        if splits[i] != 'test':
            continue
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

def prepare_for_comet(g_path, out_path):
    types = ['Precedence', 'Succession', 'Synchronous', 'Reason', 'Result', 'Condition', 'Contrast', 'Concession',
             'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception']
    columns = ['event'] + types + ['split']
    column_to_pos = dict([(v, k) for k, v in enumerate(columns)])

    g = Graph(g_path, None)

    # Note that this function only collect from G, so the contrastive negative samples are not included
    def collect_split(split, alt_split_name):
        collected_edges = defaultdict(list)
        for e in g.edges:
            if e.split == split:
                collected_edges[e.head_id].append(e)
        samples = []
        for head_id, edges in tqdm.tqdm(list(collected_edges.items())):
            line = [list() for _ in columns]
            line[0] = g.nodes[head_id].text
            line[-1] = alt_split_name
            for e in edges:
                line[column_to_pos[e.label]].append(g.nodes[e.tail_id].text)
            samples.append(line)
        cdf = pd.DataFrame(samples, columns=columns)
        cdf.iloc[:, 1:15] = cdf.iloc[:, 1:15].apply(lambda col: col.apply(json.dumps))
        return cdf

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    df_dev = collect_split('train', 'trn')
    df_dev.to_csv(os.path.join(out_path, 'v3_aser_trn.csv'), index=False, quotechar='"', doublequote='""')

    df_dev = collect_split('dev', 'dev')
    df_dev.to_csv(os.path.join(out_path, 'v3_aser_dev.csv'), index=False, quotechar='"', doublequote='""')

    df_dev = collect_split('test', 'tst')
    df_dev.to_csv(os.path.join(out_path, 'v3_aser_tst.csv'), index=False, quotechar='"', doublequote='""')

    def prepare_contrastives(c_path, alt_split_name):
        print('Build contrastives...' + alt_split_name)
        lines = open(c_path).read().splitlines()
        samples = []
        for l in tqdm.tqdm(lines):
            sent = l.split('\t')
            line = [list() for _ in columns]
            line[0] = sent[0]
            line[-1] = alt_split_name
            line[column_to_pos[sent[1]]].append(sent[2])
            samples.append(line)
        cdf = pd.DataFrame(samples, columns=list(columns))
        cdf.iloc[:, 1:15] = cdf.iloc[:, 1:15].apply(lambda col: col.apply(json.dumps))
        return cdf

    if not os.path.isdir(os.path.join(out_path, 'scoring')):
        os.makedirs(os.path.join(out_path, 'scoring'))

    df_dev = prepare_contrastives(os.path.join(g_path, 'node_contrastive_test.tsv'), 'tst')
    df_dev.to_csv(os.path.join(out_path, 'scoring', 'v3_aser_node_test.csv'), index=False, quotechar='"', doublequote='""')

    df_dev = prepare_contrastives(os.path.join(g_path, 'concept_contrastive_test.tsv'), 'tst')
    df_dev.to_csv(os.path.join(out_path, 'scoring', 'v3_aser_concept_test.csv'), index=False, quotechar='"', doublequote='""')

if __name__ == '__main__':
    KG_retrieve(r'data/aser/KG_v0.1.0.db', r'data/aser/temp/kg')
    build_graph(r'data/aser/temp/kg', r'data/aser/temp',
                r'data/probase/concepts')
    split(r'data/aser/temp', r'data/aser')
    prepare_for_comet(r'data/aser', r'data/aser/comet')