"""Classify atomic environments by nearest-neighbor coordination number."""


def coordination(neighbors_list: list[list[int]], threshold: int) -> list[str]:
    """Classify atoms as 'crystal' or 'noncrystal' based on neighbor count.

    Parameters
    ----------
    neighbors_list : list[list[int]]
        List of first nearest neighbor indices for each atom.
    threshold : int
        Minimum number of neighbors for an atom to be classified as 'crystal'.

    Returns
    -------
    list[str]
        'crystal' or 'noncrystal' classification for each atom.

    """
    result: list[str] = []
    for neighbors_i in neighbors_list:
        if len(neighbors_i) >= threshold:
            result.append("crystal")
        else:
            result.append("noncrystal")
    return result
