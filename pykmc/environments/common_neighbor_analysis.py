"""Determine cristalline environments."""

def cna_signature(neighbors_list: list[list[int]]) -> list[str] : 
    """Classify atomic environments by neighbor count.

    Determine if each atom's environment is 'crystal' (12, 8, or 6 neighbors)
    or 'noncrystal'.

    Parameters
    ----------
    neighbors_list : list[list[int]]
        List of first nearest neighbor indices for each atom.

    Returns
    -------
    list[str]
        'crystal' or 'noncrystal' classification for each atom.

    """

    all_signatures = []
    #Compute signature
    for i, neighbors_i in enumerate(neighbors_list):
        signatures = {} #signature for all i,j pairs
        for j in neighbors_i : 
            neighbors_j = neighbors_list[j] 

            #common neighbors between i and j : first signature value
            common_neighbors = list(set(neighbors_i) & set(neighbors_j)) #intersection
            n_common = len(common_neighbors)
            if n_common == 0 : 
                continue 
            
            #How many common_neighbors are first neighbors/connected 
            n_bonds = 0 
            for k in common_neighbors : 
                neighbors_k = neighbors_list[k]
                n_bonds += len(set(neighbors_k) & set(common_neighbors)) #Check neighbors of k in common neighbors of i and j
            n_bonds //=2 

            #Signature (n_common, n_bonds)
            sig = (n_common, n_bonds)
            signatures[sig] = signatures.get(sig, 0) +1 #counter of same signature
        all_signatures.append(signatures)
    return all_signatures

def cna(neighbors_list) : 
    
    all_signatures = cna_signature(neighbors_list)
    hash = []
    for signatures in all_signatures : 
        #Compute hash : 
        if is_crystal(signatures) : 
            hash.append("crystal")
        else : 
            hash.append("noncrystal")
    return hash

def is_crystal(signatures:dict) : 

    if not signatures : 
        return False
    
    #FCC : when 12 signature (4,2)
    #HCP: 6 signatures (4,2,1) and 6 signatures (4,2,2) so 12 signature (4,2)
    if signatures.get((4,2), 0) == 12 : 
        return True 
    
    #BCC : 6 signature (4,4) and 8 signature (6,6) : 
    if signatures.get((4,4), 0) == 6 and signatures.get((6,6), 0) == 8 : 
        return True 
    
    #ICO : 12 signature (5,5) 
    if signatures.get((5,5), 0) == 12 : 
        return True 
    
    return False

def cna_order_parameter(neighbors_list: list[list[int]]) -> list[float]:
    """Continuous crystallinity order parameter per atom, in the range [0, 1].

    Unlike `cna`/`is_crystal`, which require an atom's signature histogram to
    exactly match a reference structure, this scores how much overlap there is
    between the atom's signatures and the closest reference, giving a smooth
    value instead of a hard crystal/noncrystal cut.

    Parameters
    ----------
    neighbors_list : list[list[int]]
        List of first nearest neighbor indices for each atom.

    Returns
    -------
    list[float]
        Per atom, 0.0 = perfect match to a reference structure (fully
        crystalline), 1.0 = no signature overlap with any reference (fully
        non-crystalline). Values in between reflect partial/distorted order.

    """

    # Ideal (n_common, n_bonds) signature counts for each reference structure,
    # same signatures used by `is_crystal`:
    #  - fcc_hcp : 12 signatures of (4,2) (FCC and HCP share this pair signature)
    #  - bcc     : 6 signatures of (4,4) and 8 signatures of (6,6)
    #  - ico     : 12 signatures of (5,5)
    reference_histograms = {
        "fcc_hcp": {(4, 2): 12},
        "bcc": {(4, 4): 6, (6, 6): 8},
        "ico": {(5, 5): 12},
    }

    all_signatures = cna_signature(neighbors_list)
    order_parameter = []
    for signatures in all_signatures:
        if not signatures:
            # No common-neighbor signatures at all: fully disordered.
            order_parameter.append(1.0)
            continue

        # Compare the atom's signature histogram against each reference,
        # keeping the best (highest) overlap fraction found.
        best_overlap = 0.0
        for reference in reference_histograms.values():
            total = sum(reference.values())
            # Per-signature overlap: how many of the atom's counts for this
            # signature are "used up" by the reference's expected count.
            overlap = sum(
                min(signatures.get(sig, 0), count) for sig, count in reference.items()
            )
            best_overlap = max(best_overlap, overlap / total)

        # Invert so 0 = crystalline (full match) and 1 = non-crystalline (no match).
        order_parameter.append(1.0 - best_overlap)

    return order_parameter

def is_diamond(signature:dict) :

    if not signature : 
        return False 
    if signature.get((4,2),0) == 12 : 
        return True 
    return False

def identify_diamond(neighbors_list) : 
    #Neet first to have list of second neighbors atoms : 

    second_neighbors_list = [] 
    for idx, l_nei in enumerate(neighbors_list) :
        tmp = set()
        for n in l_nei : 
            tmp = tmp | set(neighbors_list[n])
        tmp.discard(idx)
        second_neighbors_list.append(tmp)

    #then compute cna signature
    all_signatures = cna_signature(second_neighbors_list)
    hash = []
    #check is fcc or bcc
    for signatures in all_signatures : 
        if is_diamond(signatures) : 
            hash.append("crystal")
        else : 
            hash.append("noncrystal")

    return hash






