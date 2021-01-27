import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from itertools import combinations
from pymongo import MongoClient
from multiprocessing import Pool
import time


class CosineLSH(object):
    def __init__(self, n_vectors, dim, num_tables=100, db_name='test'):
        self.client = MongoClient()
        self.db_name = db_name
        if db_name in self.client.list_database_names():
            self.num_tables = self.client[db_name]['base_vectors'].count_documents({
            })
            self.base_vectors = []
            for idx in range(num_tables):
                r = self.client[db_name].base_vectors.find_one({"table": idx})
                self.base_vectors.append(r['base_vector'])
            self.base_vectors = [np.array(r['base_vector'])
                                 for idx in range(self.num_tables)]
            self.dim = self.base_vectors[0].shape[1]
            self.n_vectors = self.base_vectors[0].shape[0]
            self.tables = [self.client[db_name]
                           [f'table_{i}'] for i in range(num_tables)]
        else:
            self.base_vectors = [np.random.randn(
                n_vectors, dim) for i in range(num_tables)]
            self.client[db_name]['base_vectors'].insert_many(
                [{'table': idx, f'base_vector': v.tolist()} for idx, v in enumerate(self.base_vectors)])
            self.dim = dim
            self.n_vectors = n_vectors
            self.num_tables = num_tables
            self.tables = [self.client[db_name]
                           [f'table_{i}'] for i in range(num_tables)]
            for table in self.tables:
                table.create_index("bucket")
            self.client[db_name]['bases'].create_index('index')

    def insert(self, table, index, names):
        current_record = table.find_one({"bucket": index})
        if type(current_record) != None:
            names = current_record['names'] + names
            table.find_one_and_update({"_id": current_record['_id']}, {
                                      "$set": {"names": names}})
        else:
            table.insert_one({"bucket": int(index), "names": names})

    def index_one(self, vector, name):
        for base_vector, table in zip(self.base_vectors, self.tables):
            index = vector.dot(base_vector.T) > 0
            index = (2**np.array(range(self.n_vectors)) * index).sum()
            self.insert(table, index, name)

    def index_batch(self, vectors, names):
        for base_vector, table in tqdm(zip(self.base_vectors, self.tables)):
            temp_table = defaultdict(list)
            indices = vectors.dot(base_vector.T) > 0
            indices = indices.dot(
                2 ** np.array(range(self.n_vectors))).astype(int)
            for index, name in zip(indices, names):
                temp_table[index].append(name)
            records = [{'bucket': int(_index), "names": _name}
                       for _index, _name in temp_table.items()]
            table.insert_many(records)

    def query(self, vector, radius=1):
        res = []
        blocks = []
        for table_idx, base_vector, table in zip(range(self.num_tables), self.base_vectors, self.tables):
            index = vector.dot(base_vector.T) > 0
            _index = [int((2**np.array(range(self.n_vectors)) * index).sum())]
            for _r in range(1, radius+1):
                for _reverse_idx in combinations(range(self.n_vectors), _r):
                    nearby_index = index.copy()
                    nearby_index[list(_reverse_idx)] = np.logical_not(
                        nearby_index[list(_reverse_idx)])
                    nearby_index = (
                        2**np.array(range(self.n_vectors)) * nearby_index).sum()
                    _index.append(int(nearby_index))
            blocks.append([self.db_name, table_idx, _index])
        for block in blocks:
            res += query_table(block)
        return res
