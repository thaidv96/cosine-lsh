import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from iteration_utilities import duplicates, flatten, unique_everseen


class CosineLSH(object):

    def __init__(self, n_vectors, dim, num_tables=100):
        self.n_vectors = n_vectors
        self.base_vectors = [np.random.randn(
            n_vectors, dim) for i in range(num_tables)]
        self.base_vector = np.vstack(self.base_vectors)
        self.num_tables = num_tables
        self.hash_table = np.empty((2**n_vectors * num_tables,), object)
        self.hash_table[...] = [[] for _ in range(2**n_vectors * num_tables)]
        self.dim = dim
        self.cal_index_times = []
        self.lookup_index_times = []
        self.flatten_times = []
        self.vectors = None
        self.current_idx = 0
        self.names = []

    def index_one(self, vector, name):
        for hash_table_idx, base_vector in enumerate(self.base_vectors):
            index = vector.dot(base_vector.T) > 0
            index = (2**np.array(range(self.n_vectors)) * index).sum()
            relative_index = hash_table_idx * 2 ** self.n_vectors + index
            self.hash_table[relative_index].append(self.current_idx)
        self.names.append(name)

            
        if type(self.vectors) == type(None):
            self.vectors = vector
        else:
            self.vectors = np.vstack([self.vectors, vector])
    
    def index_batch(self, vectors, names):
        idxs = range(self.current_idx, self.current_idx+ vectors.shape[0])
        for hash_table_idx, base_vector in tqdm(enumerate(self.base_vectors), total = self.num_tables):
            indices = vectors.dot(base_vector.T) > 0
            indices = indices.dot(2 ** np.array(range(self.n_vectors)))
            for index, idx in zip(indices, idxs):
                relative_index = hash_table_idx * 2 ** self.n_vectors + index
                self.hash_table[relative_index].append(idx)
        self.current_idx += vectors.shape[0]
        self.names += names
        if type(self.vectors) == type(None):
                self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

       
    def query(self, vector, radius=1,top_k=5):
        res_indices = []
        ## Need to improve index calculations
        indices = vector.dot(self.base_vector.T).reshape(self.num_tables,-1) > 0
        if radius == 0:
            res_indices = indices.dot(2**np.arange(self.n_vectors)) + np.arange(self.num_tables) * 2**self.n_vectors
        elif radius == 1:
            clone_indices = indices.repeat(axis=0,repeats= self.n_vectors)
            rel_indices = (np.arange(self.num_tables) * 2**self.n_vectors).repeat(axis=0,repeats=self.n_vectors)
            translate = np.tile(np.eye(self.n_vectors), (self.num_tables,1))
            res_indices = (np.abs(clone_indices-translate).dot(2**np.arange(self.n_vectors)) + rel_indices).astype(int)
            res_indices = np.concatenate([res_indices, indices.dot(2**np.arange(self.n_vectors)) + np.arange(self.num_tables) * 2**self.n_vectors])
    
        start = time.time()
        lst = self.hash_table[res_indices].tolist()
        self.lookup_index_times.append(time.time() - start)
        start = time.time()

        res = list(unique_everseen(duplicates(flatten(lst))))
        sim_scores = vector.dot(self.vectors[res].T)

        max_sim_indices = sim_scores.argsort()[-top_k:][::-1]
        max_sim_scores = sim_scores[max_sim_indices]

        return [(self.names[res[i]], score) for i, score in zip(max_sim_indices, max_sim_scores)]