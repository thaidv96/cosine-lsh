import numpy as np

from lsh_cist import CosineLSH

## Prepare dataset (feature vectors and names)
vecs = np.random.randn(10000, 512)
norms = np.linalg.norm(vecs, axis=1)
vecs = vecs/norms.reshape(-1,1)
names = range(10000)


# Initialize lsh object
lsh = CosineLSH(16, 512, 200)

# Create index for dataset

lsh.index_batch(vecs, names)

## Test query:

v = vecs[1231]

result = lsh.query(v, top_k = 10)
print(result)


