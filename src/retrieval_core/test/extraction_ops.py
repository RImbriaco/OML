import torch


def unpack_vectors(vec_list):
    return torch.cat([vl[0] for vl in vec_list], dim=1)


def extract_ss(logits_pack, triplet_pack):
    """
    Extract embeddings at a single scale.
    :param logits_pack: Obtained from the branches that employ
    classification losses.
    :param triplet_pack: Obtained from branches using metric losses.
    :return:
    Image embeddings.
    """
    logits_pack = unpack_vectors(logits_pack)
    triplet_pack = unpack_vectors(triplet_pack)
    vec = torch.cat((logits_pack, triplet_pack), dim=1).data
    return vec / torch.norm(vec, dim=1).unsqueeze(-1)

