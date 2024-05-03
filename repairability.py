import numpy as np


def repairability_degree(og_score, repaired_score):
    """
    The performance score must be comprised between 0 and 1 with 1 the best performance and 0 the worst.
    :param og_score: original performance score before repairing (we propose to use 1-qa1)
    :param repaired_score: performance score after the repairing pipeline was applied to the dataset
    :return: the degree of repairability of the dataset for the repairing pipeline
    """
    if np.isnan(repaired_score):
        return np.nan
    delta = repaired_score - og_score
    gain_max = 1 - og_score
    det_max = og_score
    lim = gain_max
    if delta < 0:
        lim = det_max
    deg_rep = delta / lim
    return deg_rep
