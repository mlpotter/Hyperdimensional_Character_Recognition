# Press the green button in the gutter to run the script.
import numpy as np

def load_small_mnist():
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    digits = load_digits()
    digits_data = digits['data']
    digits_data_rounded = (digits_data > 7.5).astype(np.int8)
    target = digits['target']

    plt.gray()
    plt.matshow(digits_data_rounded[0].reshape(8,8))
    plt.show()

    return (digits_data_rounded,target)

def load_large_mnist():
    from sklearn.datasets import fetch_openml
    import matplotlib.pyplot as plt
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    digits_data_rounded = (X > 128).astype(np.int8)
    target = y.astype(np.int8)

    plt.gray()
    plt.matshow(digits_data_rounded[0].reshape(28,28))
    plt.show()

    return (digits_data_rounded,target)