import numpy as np
from scipy.spatial.distance import cdist


def jaccard_mining(sim_threshold, target, embeddings):
    """
    Select hard postive and negative candidates from a list of embeddings based
    on the label similarity.
    :param sim_threshold: Threshold determines positives.
    :param target: One-hot encoded labels.
    :param embeddings: Image embeddings.
    :return:
    Anchor, positive and negative triplets.
    """
    target_np = target.cpu().numpy()
    dist = cdist(target_np, target_np, 'jaccard')
    tresh_dist = dist <= sim_threshold
    neg_pool = list(range(len(dist)))

    pos_cand = [np.nonzero(i)[0] for i in tresh_dist]
    neg_cand = [list(set(neg_pool).symmetric_difference(pc)) for pc in
                pos_cand]

    pos_idx = [np.random.choice(pc) for pc in pos_cand]
    neg_idx = [np.random.choice(nc) for nc in neg_cand]
    anchor = embeddings
    pos = embeddings[pos_idx]
    neg = embeddings[neg_idx]
    return anchor, pos, neg
