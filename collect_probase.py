import tqdm
import pickle
probase = open('data/probase/data-concept-instance-relations.txt')
concepts = set()
for line in tqdm.tqdm(probase):
    line = line.split('\t')
    concepts.add(line[0])
    concepts.add(line[1])
pickle.dump(concepts, open('data/probase/concepts', 'wb'))