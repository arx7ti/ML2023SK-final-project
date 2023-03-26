import torch
import numpy as np

# Utils copied and slightly modified from https://github.com/sambaiga/MLCFCD


def get_distance_measure(x: torch.Tensor, p: int = 1):
    """Given input Nxd input compute  NxN  distance matrix, where dist[i,j]
        is the square norm between x[i,:] and x[j,:]
        such that dist[i,j] = ||x[i,:]-x[j,:]||^p]]
    
    Arguments:
        x {torch.Tensor} -- [description]
    
    Keyword Arguments:
        p {int} -- [description] (default: {1})
    
    Returns:
        [dist] -- [NxN  distance matrix]
    """

    N, D = x.size()
    dist = torch.repeat_interleave(x, N, dim=1)
    dist.permute(1, 0)
    dist = torch.pow(torch.abs(dist - dist.permute(1, 0))**p, 1 / p)

    return dist


def paa(series: np.array, emb_size: int, scaler=None):
    """
    Piecewise Aggregate Approximation (PAA)  a dimensionality reduction 
      method for time series signal based on saxpy.
      https://github.com/seninp/saxpy/blob/master/saxpy/paa.py
    
    Arguments:
        series {np.array} -- [NX1 input series]
        emb_size {int} -- [embedding dimension]
    
    Returns:
        [series] -- [emb_size x 1]
    """

    series_len = len(series)
    if scaler:
        series = series / scaler

    # check for the trivial case
    if (series_len == emb_size):
        return np.copy(series)
    else:
        res = np.zeros(emb_size)
        # check when we are even
        if (series_len % emb_size == 0):
            inc = series_len // emb_size
            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res, idx, series[i])
                # res[idx] = res[idx] + series[i]
            return res / inc
        # and process when we are odd
        else:
            for i in range(0, emb_size * series_len):
                idx = i // series_len
                pos = i // emb_size
                np.add.at(res, idx, series[pos])
                # res[idx] = res[idx] + series[pos]
            return res / series_len


def fryze_power_decomposition(i, v):
    p = i * v
    vrsm = v**2
    i_active = p.mean() * v / vrsm.mean()
    i_non_active = i - i_active
    return i_active, i_non_active


def compute_active_non_active_features(current, voltage, emb_size=50):
    n = len(current)
    # with tqdm(n) as pbar:
    features = []
    for k in range(n):
        i_active, i_non_active = fryze_power_decomposition(
            current[k], voltage[k])
        if emb_size < len(current[k]):
            i_active = paa(i_active.flatten(), emb_size)
            i_non_active = paa(i_non_active.flatten(), emb_size)
        else:
            i_active = i_active.flatten()
            i_non_active = i_non_active.flatten()
        features.append(np.hstack([i_non_active[:, None], i_active[:, None]]))
        # pbar.set_description('frze processed: %d percent' % round(
        #     (1 + k) * 100 / n, 2))
        # pbar.update(1)
    # pbar.close()
    features = torch.tensor(features).float().transpose(1, 2)
    return features


def compute_similarities_distance(current, p):
    dist = []
    for k in range(len(current)):
        dist += [get_distance_measure(current[k].unsqueeze(1), p=p)]

    return torch.stack(dist)


def generate_input_feature(current, voltage, width=50):
    feature = compute_active_non_active_features(current, voltage, width)
    dist_1 = compute_similarities_distance(feature[:, 0, :], 2).unsqueeze(1)
    dist_2 = compute_similarities_distance(feature[:, 1, :], 2).unsqueeze(1)
    feature = torch.cat([dist_1, dist_2], 1)
    return feature