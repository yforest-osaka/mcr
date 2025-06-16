from tqdm import tqdm
import numpy as np
from mcr.gate_apply import PauliBit
from mcr.gate_apply import (
    zhang_optimization,
    grouping,
    loop_optimization,
    set_clifford_to_qulacs,
)
from mcr.mcr_optimize import find_mcr, find_nontrivial_swap
from mcr.equiv_check import equiv
from more_itertools import distinct_permutations
from joblib import Parallel, delayed
from opt_using_mcr import test_algorithm


def flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def check_equiv(i, elem, base_flat):
    elem_flat = flatten(elem)
    if len(elem_flat) != len(base_flat):
        return None
    if equiv([[], elem_flat], [[], base_flat]):
        if len(test_algorithm(elem_flat, show_opt_log=False)[1]) == 0:
            return i
    return None


def main():
    base = [
        # PauliBit("IZ", -np.pi / 4),
        # PauliBit("ZY", np.pi / 4),
        PauliBit("XY", -np.pi / 4),
        PauliBit("IZ", -np.pi / 4),
        PauliBit("XZ", -np.pi / 4),
        PauliBit("IY", -np.pi / 4),
        PauliBit("YZ", np.pi / 4),
        PauliBit("IX", -np.pi / 4),
        PauliBit("ZI", np.pi / 4),
        PauliBit("YY", -np.pi / 4),
        PauliBit("XI", -np.pi / 4),
        PauliBit("ZI", -np.pi / 4),
        PauliBit("ZZ", np.pi / 4),
        PauliBit("XI", np.pi / 4),
        # PauliBit("ZX", np.pi / 4),
        # PauliBit("XX", np.pi / 4),
    ]
    base_flat = flatten(base)

    seq_a = [
        # [PauliBit("IZ", -np.pi / 4)],
        # [PauliBit("ZY", np.pi / 4)],
        [PauliBit("XY", -np.pi / 4)],
        [PauliBit("IZ", -np.pi / 4), PauliBit("XZ", -np.pi / 4)],
        [PauliBit("IY", -np.pi / 4)],
        [PauliBit("YZ", np.pi / 4)],
        [PauliBit("IX", -np.pi / 4), PauliBit("ZI", np.pi / 4)],
        [PauliBit("YY", -np.pi / 4)],
        [PauliBit("XI", -np.pi / 4)],
        [PauliBit("ZI", -np.pi / 4), PauliBit("ZZ", np.pi / 4)],
        [PauliBit("XI", np.pi / 4)],
        # [PauliBit("ZX", np.pi / 4)],
        # [PauliBit("XX", np.pi / 4)],
    ]

    perms = list(distinct_permutations(seq_a))

    results = Parallel(n_jobs=10)(
        delayed(check_equiv)(i, elem, base_flat)
        for i, elem in tqdm(
            enumerate(perms), total=len(perms), desc="Checking equivalence"
        )
    )

    # フィルタリングと保存
    results = list(filter(lambda x: x is not None, results))
    with open("results_2.txt", "w") as f:
        for i in results:
            f.write(f"{i}\n")

    # for i, elem in tqdm(
    #     enumerate(perms), total=len(perms), desc="Checking equivalence"
    # ):
    #     if i in {264}:
    #         print(elem)
    #         print("-------------------")


if __name__ == "__main__":
    main()
