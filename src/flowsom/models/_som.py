from __future__ import annotations

import numpy as np
from numba import jit, prange
from numbasom import SOM as numbaSOM


@jit(nopython=True, parallel=False)
def eucl(p1, p2):
    distance = 0.0
    for j in range(len(p1)):
        diff = p1[j] - p2[j]
        distance += diff * diff
    return np.sqrt(distance)

@jit(nopython=True, parallel=False)
def eucl_sq(p1, p2):
    distance = 0.0
    for j in range(len(p1)):
        diff = p1[j] - p2[j]
        distance += diff * diff
    return distance


@jit(nopython=True, parallel=False)
def eucl_sq2(p1, p2):
    diff = p1 - p2
    return np.dot(diff, diff)


@jit(nopython=True, parallel=True)
def manh(p1, p2):
    return np.sum(np.abs(p1 - p2))


@jit(nopython=True)
def chebyshev(p1, p2, px, n, ncodes):
    distance = 0.0
    for j in range(px):
        diff = abs(p1[j * n] - p2[j * ncodes])
        if diff > distance:
            distance = diff
    return distance


@jit(nopython=True, parallel=True)
def cosine(p1, p2, px, n, ncodes):
    nom = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for j in range(px):
        nom += p1[j * n] * p2[j * ncodes]
        denom1 += p1[j * n] * p1[j * n]
        denom2 += p2[j * ncodes] * p2[j * ncodes]

    return (-nom / (np.sqrt(denom1) * np.sqrt(denom2))) + 1


def SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None, version="batch2_sq"):
    if version == "original":
        return original_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf, seed)
    elif version == "squared":
        return original_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen,
                            eucl_sq, # use squared euclidean distance
                            seed)
    elif version == "numba":
        return numba_SOM(data, codes, rlen, seed)

    elif version == "batch":
        return batch_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, seed=42, batch_size=50)

    elif version == "batch2_sq":
        return batch_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, batch_size=50, distf=eucl_sq, seed=42)

    elif version == "batch2_sq2":
        return batch_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, batch_size=50, distf=eucl_sq2, seed=42)

    elif version == "batch2_eucl":
        return batch_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, batch_size=50, distf=eucl, seed=42)



@jit(nopython=True, parallel=True)
def map_data_to_codes(data, codes, distf=eucl):
    counter = -1
    n_codes = codes.shape[0]
    nd = data.shape[0]
    nn_codes = np.zeros(nd)
    nn_dists = np.zeros(nd)
    for i in range(nd):
        minid = -1
        mindist = np.inf
        for cd in range(n_codes):
            tmp = distf(data[i, :], codes[cd, :])
            if tmp < mindist:
                mindist = tmp
                minid = cd
        counter += 1
        nn_codes[counter] = minid
        nn_dists[counter] = mindist
    return nn_codes, nn_dists


@jit(nopython=True, parallel=True)
def original_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None):
    if seed is not None:
        np.random.seed(seed)
    xdists = np.zeros(ncodes)
    n = data.shape[0]
    px = data.shape[1]
    niter = rlen * n
    threshold = radii[0]
    thresholdStep = (radii[0] - radii[1]) / niter
    change = 1.0

    for k in range(niter):
        if k % n == 0:
            if change < 1:
                k = niter
            change = 0.0

        i = np.random.randint(n)

        nearest = 0
        for cd in range(ncodes):
            xdists[cd] = distf(data[i, :], codes[cd, :])
            if xdists[cd] < xdists[nearest]:
                nearest = cd

        if threshold < 1.0:
            threshold = 0.5
        alpha = alphas[0] - (alphas[0] - alphas[1]) * k / niter

        for cd in range(ncodes):
            if nhbrdist[cd, nearest] > threshold:
                continue

            for j in range(px):
                tmp = data[i, j] - codes[cd, j]
                change += abs(tmp)
                codes[cd, j] += tmp * alpha

        threshold -= thresholdStep
    return codes

def numba_SOM(data, ncodes, rlen, seed=None):
    """Calculate the SOM using the numbaSOM implementation.

    This code comes from the public GitHub repository https://github.com/nmarincic/numbasom by user nmarincic.
    The parameters are the same as the SOM function in this file.
    The numbaSOM implementation is a bit faster than the original implementation but offers less flexibility.
    """
    if seed is not None:
        np.random.seed(seed)

    xdim = int(np.sqrt(ncodes))
    n = data.shape[0]

    iterations = rlen * n
    numbasom = numbaSOM(som_size=(xdim, xdim), is_torus=False)

    lattice = numbasom.train(data, iterations)

    # the lattice reshaped to the shape
    codes = lattice.reshape((ncodes, data.shape[1]))
    return codes


@jit(nopython=True, parallel=True)
def batch_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, seed=None, batch_size=10):
    """
    Optimized Batch SOM algorithm with mini-batch training and precomputed neighborhood influence.

    Args:
        data (np.ndarray): The dataset to be clustered.
        codes (np.ndarray): The SOM codebook (initially random or pre-trained).
        nhbrdist (np.ndarray): Neighborhood distance matrix.
        alphas (tuple): The starting and ending learning rates.
        radii (tuple): The starting and ending neighborhood radii.
        ncodes (int): Number of codes (nodes) in the SOM.
        rlen (int): Number of iterations to train the SOM.
        seed (int): Random seed for reproducibility.
        batch_size (int): The size of the mini-batches for training.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize dimensions and precompute threshold adjustments
    n, px = data.shape
    niter = rlen * n
    threshold = radii[0]
    threshold_step = (radii[0] - radii[1]) / niter
    alpha = alphas[0]
    alpha_step = (alphas[0] - alphas[1]) / niter

    # Precompute the neighborhood influence for each pair of nodes
    influence = np.exp(-nhbrdist / (2 * (threshold ** 2)))

    # Mini-batch training loop
    for _ in range(rlen):
        # Shuffle data
        indices = np.random.permutation(n)
        for batch_start in range(0, n, batch_size):
            # Extract mini-batch
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_data = data[batch_indices, :]

            # Compute distances between the batch and all codes
            distances = np.empty((batch_size, ncodes))
            for i in prange(batch_size):
                for cd in prange(ncodes):
                    distances[i, cd] = np.linalg.norm(batch_data[i, :] - codes[cd, :])

            # Find the BMUs (Best Matching Units) for each data point in the batch
            bmus = np.argmin(distances, axis=1)

            # Update the codebook using the mini-batch
            for i in prange(batch_size):
                bmu = bmus[i]
                # Influence adjustment
                for cd in prange(ncodes):
                    if influence[cd, bmu] > threshold:
                        codes[cd, :] += alpha * influence[cd, bmu] * (batch_data[i, :] - codes[cd, :])

        # Adjust learning rate and threshold
        alpha -= alpha_step
        threshold -= threshold_step

        # Recompute influence based on new threshold
        influence = np.exp(-nhbrdist / (2 * (threshold ** 2)))

    return codes


@jit(nopython=True, parallel=True)
def batch2_SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, batch_size=0, distf=eucl_sq, seed=None):
    """
    Optimized Batch SOM with consistent arguments.

    Args:
        data (np.ndarray): Input data for training the SOM.
        codes (np.ndarray): Initial weight vectors (SOM nodes).
        nhbrdist (np.ndarray): Precomputed neighborhood distances between SOM nodes.
        alphas (tuple): Start and end learning rates.
        radii (tuple): Start and end radii for neighborhood influence.
        ncodes (int): Number of SOM nodes.
        rlen (int): Number of epochs (iterations) for training.
        batch_size (int): Number of data points per batch (default is 0, meaning full dataset).
        distf (function): Distance function to compute similarity (default is Euclidean).
        seed (int): Random seed for reproducibility (default is None).

    Returns
    -------
        np.ndarray: Updated SOM nodes (codes) after training.
    """
    # Ensure reproducibility if seed is provided
    if seed is not None:
        np.random.seed(seed)

    # Handle batch size defaulting to full dataset if not specified
    n_samples = data.shape[0]
    if batch_size == 0:
        batch_size = n_samples

    # Initialize training parameters
    n_batches = n_samples // batch_size
    n_iterations = rlen * n_batches
    current_radius = radii[0]
    radius_decay = (radii[0] - radii[1]) / n_iterations
    start_lr = alphas[0]
    end_lr = alphas[1]

    # Main training loop across iterations
    for iteration in range(n_iterations):
        # Randomly select a batch of data points
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        batch_data = data[batch_indices]

        # Initialize accumulators for weight updates
        node_hits = np.zeros(ncodes)
        update_accumulator = np.zeros_like(codes)

        # Iterate over the batch
        for b_idx in prange(batch_size):
            data_point = batch_data[b_idx]

            # Compute distances from the data point to all SOM nodes
            distances = np.zeros(ncodes)
            for node_idx in prange(ncodes):
                distances[node_idx] = distf(data_point, codes[node_idx])

            # Find the Best Matching Unit (BMU)
            bmu_idx = np.argmin(distances)

            # Update the neighborhood of the BMU
            for node_idx in prange(ncodes):
                if nhbrdist[node_idx, bmu_idx] <= current_radius:
                    node_hits[node_idx] += 1
                    update_accumulator[node_idx] += (data_point - codes[node_idx])

        # Update learning rate
        learning_rate = start_lr - (start_lr - end_lr) * iteration / n_iterations

        # Apply the accumulated updates to the SOM nodes
        for node_idx in prange(ncodes):
            if node_hits[node_idx] > 0:
                update_accumulator[node_idx] /= node_hits[node_idx]  # Normalize updates by the number of hits
                codes[node_idx] += learning_rate * update_accumulator[node_idx]

        # Decay the neighborhood radius
        current_radius -= radius_decay

    return codes
