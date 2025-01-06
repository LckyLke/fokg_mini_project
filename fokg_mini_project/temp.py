class KG:
    def __init__(self, data_dir=None):
        
        # 1. Parse the benchmark dataset
        s = '------------------- Description of Dataset' + data_dir + '----------------------------'
        print(f'\n{s}')
        self.train = self.load_data(data_dir + 'train.txt')
        self.valid = self.load_data(data_dir + 'valid.txt')
        self.test = self.load_data(data_dir + 'test.txt')
        
        self.all_triples = self.train + self.valid + self.test
        self.entities = self.get_entities(self.all_triples)
        self.relations = self.get_relations(self.all_triples)

        # 2. Index entities and relations
        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}

        print(f'Number of triples: {len(self.all_triples)}')
        print(f'Number of entities: {len(self.entities)}')
        print(f'Number of relations: {len(self.relations)}')
        print(f'Number of triples on train set: {len(self.train)}')
        print(f'Number of triples on valid set: {len(self.valid)}')
        print(f'Number of triples on test set: {len(self.test)}')
        s = len(s) * '-'
        print(f'{s}\n')

        # 3. Index train, validation and test sets 
        self.train_idx = [(self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]) for s, p, o in self.train]
        self.valid_idx = [(self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]) for s, p, o in self.valid]
        self.test_idx = [(self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]) for s, p, o in self.test]

        # 4. Create mappings for the filtered link prediction
        self.sp_vocab = dict()
        self.po_vocab = dict()
        self.so_vocab = dict()

        for i in self.all_triples:
            s, p, o = i[0], i[1], i[2]
            s_idx, p_idx, o_idx = self.entity_idxs[s], self.relation_idxs[p], self.entity_idxs[o]
            self.sp_vocab.setdefault((s_idx, p_idx), []).append(o_idx)
            self.so_vocab.setdefault((s_idx, o_idx), []).append(p_idx)
            self.po_vocab.setdefault((p_idx, o_idx), []).append(s_idx)


    @staticmethod
    def load_data(data_dir):
        with open(data_dir, "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    @property
    def num_entities(self):
        return len(self.entities)
    
    @property
    def num_relations(self):
        return len(self.relations)