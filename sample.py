from itertools import permutations
from tqdm import tqdm
import numpy as np
from qulacs import QuantumCircuit
from itertools import combinations
from mcr.equiv_check import (
    equivalence_check_via_mqt_qcec,
    equiv,
    pauli_bit_equivalence_check,
)
from mcr.gate_apply import (
    PauliBit,
    grouping,
    loop_optimization,
    set_clifford_to_qulacs,
    zhang_optimization,
    multiply_all,
)
import pickle

from mcr.mcr_optimize import find_nontrivial_swap, find_mcr
from opt_using_mcr import test_algorithm, mcr_swap, three_layer_nontrivial_swap
from joblib import Parallel, delayed


def main():
    nqubits = 2
    with open(f"unopt_{nqubits}.pickle", "rb") as f:
        seq = pickle.load(f)
    data = []
    for elem in seq:
        sgn = str(elem[1])[0]
        pauli_str = str(elem[1])[1:]
        if sgn == "+":
            data.append(PauliBit(pauli_str, np.pi / 4))
        else:
            assert sgn == "-", f"Unexpected sign: {sgn}"
            data.append(PauliBit(pauli_str, -np.pi / 4))
    data.append(PauliBit("Z" * nqubits, -np.pi / 4))  # Add identity gate

    seq_a = [PauliBit(ele.get_pauli_str(), ele.get_angle()) for ele in data]
    clifford_lst, optimized_data = test_algorithm(seq_a, show_opt_log=True)

    # 何回かループさせないと消せないサンプルが存在する(concatenationと関係？)

    tmp = grouping(optimized_data)

    def process_perm(perm):
        target = sum(perm, [])
        cli, tmp2 = test_algorithm(target, show_opt_log=False)
        if equiv([cli, tmp2], [[], optimized_data]) and len(tmp2) < len(target):
            print("New length:", len(tmp2))
            return tmp2
        return None

    perms = list(permutations(tmp))
    results = Parallel(n_jobs=5)(delayed(process_perm)(perm) for perm in tqdm(perms))
    results = [r for r in results if r is not None]


if __name__ == "__main__":
    main()
