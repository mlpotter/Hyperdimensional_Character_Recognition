import numpy as np
from operations import *
from tqdm import tqdm

def encode_image(X,base_HDVs):
    # use linear indexing for the base hyperdimensional vectors
    X = X.ravel()
    positives = permutation(base_HDVs[np.where(X==1)])
    negatives = base_HDVs[np.where(X==0)]
    HDVs = np.vstack((positives,negatives))

    H = bundling(HDVs)


    return H

def encode_dataset(X,base_HDVs):
    return np.array([encode_image(x,base_HDVs) for x in tqdm(X)])

def encode_class_HDVs(X_HDVs,y):
    y_unique = np.unique(y)

    # iterate through each class
    class_HDVs = []

    for c in y_unique:
        # bundle all HDVs for the specific class to generate the class HDVc
        subset = X_HDVs[y==c]
        print(f"Class {c}: ",subset.shape)
        class_HDVs.append(bundling(subset))

    return np.vstack(class_HDVs)

def encode_image_batch(X,base_HDVs,batch_size=128):
    # use linear indexing for the base hyperdimensional vectors
    X = np.expand_dims(X,-1)
    sum_axis = max(X.ndim - 2,0)

    batches = int(np.ceil(X.shape[0]/batch_size))
    HDVs = []
    for i in tqdm(range(batches)):
        X_batch = X[(i*batch_size):(i+1)*batch_size]
        base_HDVs_rep = np.tile(base_HDVs,(X_batch.shape[0],1,1))
        HDVs.append(np.mean(X_batch * permutation(base_HDVs_rep) + (1-X_batch) * base_HDVs_rep,axis=sum_axis))

    H = np.round(np.vstack(HDVs)/X.shape[0]).astype(np.int8)

    return H

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_HDVs = generate_HDV(1000,784)
    X = (np.random.rand(28,28)>.5).astype(int)

    H = encode_image(X,base_HDVs)
    print(H)