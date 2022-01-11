# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def bundling(X):
    # Use a breakpoint in the code line below to debug your script.
    sum_axis = max(X.ndim - 2,0)
    z = np.mean(X,axis=sum_axis)
    z = np.round(z).astype(np.int8)
    return z

def association(x,y):
    return np.logical_xor(x,y).astype(np.int8)

def permutation(x,p=1):
    return np.roll(x,p,axis=-1)

def distance(x,y):
    return np.sum(x != y)

def generate_HDV(d,N=1):
    return (np.random.rand(N,d) > .5).astype(np.int8)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_HDVs = generate_HDV(5,3)

    x = base_HDVs[0:1,:]
    y = base_HDVs[1:2,:]
    z = base_HDVs[2:3,:]

    bundled = bundling(base_HDVs)
    associated = association(y,z)
    perm = permutation(z)

    dist = distance(x,y)

    print("{:15s}".format("x: "),x)
    print("{:15s}".format("y: "),y)
    print("{:15s}".format("z: "),z)
    print("{:15s}".format("perm(z): "),perm)
    print("{:15s}".format("x+y+z: "),bundled)
    print("{:15s}".format("y*z: "),associated)
    print("{:15s}".format("hamming(x,y): "),dist)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/