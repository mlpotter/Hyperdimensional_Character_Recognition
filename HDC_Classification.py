from encoders import *
from operations import *
import numpy as np
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from datasets import *
from tqdm import tqdm


def create_predictions(HDVs,class_HDVs,batch_size=128):
    class_HDVs = np.array(class_HDVs)
    HDVs = np.array(HDVs)


    batches = int(np.ceil(HDVs.shape[0]/batch_size))

    predictions = []
    for i in tqdm(range(batches)):
        HDV_batch = HDVs[(i*batch_size):(i+1)*batch_size]
        hamming_distances = metrics.pairwise_distances(HDV_batch, class_HDVs, distance)
        predictions.extend(hamming_distances.argmin(1))

    return np.array(predictions,dtype=np.int8)



if __name__ == '__main__':
    X,y = load_small_mnist()

    D = 10000
    C = len(np.unique(y))
    B = X.shape[1]

    base_HDVs = generate_HDV(D,B)
    print("{:15s}".format("# Base HDVs: "),B)
    print("{:15s}".format("Base HDV Size: "),D)
    print("{:15s}".format("# Class HDVs: "),C)
    print(X.shape)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.6, shuffle=False
    )

    X_train_HDVs = encode_dataset(X_train,base_HDVs)
    X_test_HDVs = encode_dataset(X_test,base_HDVs)

    class_HDVs = encode_class_HDVs(X_train_HDVs,y_train)

    y_pred = create_predictions(X_test_HDVs,class_HDVs)

    print(y_pred.shape)
    print(y_test.shape)

    print(metrics.classification_report(y_test,y_pred))