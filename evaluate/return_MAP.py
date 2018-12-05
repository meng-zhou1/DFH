import numpy as np

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcMap(qB, rB, queryL, retrievalL):

    if qB.dtype == 'bool':
        qB = qB + 0
    if rB.dtype == 'bool':
        rB = rB + 0
    num_query = queryL.shape[0]
    map = 0

    queryL = label_encode(queryL)
    retrievalL = label_encode(retrievalL)

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query

    return map

def label_encode(y):

    labels = np.unique(y)
    code = np.zeros((y.shape[0], labels.shape[0]))

    for i in range(code.shape[0]):
        index = np.where(labels == y[i])[0]
        code[i, index] = 1

    return code
