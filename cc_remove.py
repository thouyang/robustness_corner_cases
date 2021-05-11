import numpy as np
import scipy.io as scio
import os
from imutils import paths
from collections import Counter
import argparse

##### corner case remove and non-cc index
def cc_index(d_length,k):
    test_paths = list(paths.list_files(os.getcwd() + "/results", contains='dsa'))
    h_dsa = []
    nc_idx=np.array(range(d_length))

    for i, path in enumerate(test_paths):
        dsa=scio.loadmat(path)['dsa'][0]
        ind=np.argsort(-dsa)
        h_dsa=np.concatenate((h_dsa, ind[:k]), axis=0)

    h_dsa=h_dsa.tolist()
    tmp =np.array( Counter(h_dsa).most_common(k))
    cc_idx=np.int32(tmp[:,0])
    nc_idx = np.delete(nc_idx, cc_idx)
    return nc_idx

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlength", "-dlength", help="length of training data", type=int, default=60000)
    parser.add_argument("--k", "-k", help="number of corner cases", type=int, default=200)
    args=parser.parse_args()
    nc_idx=cc_index(d_length=args.dlength,k=args.k)
    mdic={'nc':nc_idx}
    scio.savemat('results/nc_idx.mat',mdic)