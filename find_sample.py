from mcr.gate_apply import PauliBit
from mcr.equiv_check import (
    equiv,
)
from qulacs import QuantumCircuit
from mcr.gate_apply import set_clifford_to_qulacs
from time import time
import numpy as np
import pickle
from mcr.rotation_circuit import PauliRotationSequence
from mcr.unoptimize import unoptimize_circuit
from tqdm import tqdm
from copy import deepcopy
from optimizer import full_optimization
from typing import List, Tuple


def main():
    filetype = "seq"  # "small" or "seq"
    num_samples = 1000
    nqubits = 3
    with_swap_option = True  # If True, the MCR swap is executed (then the unoptimized circuit becomes longer)
    # Number of iterations for the unoptimized circuit
    unopt_iteration_count = 3
    for _ in tqdm(
        range(num_samples), desc=f"Processing circuits with {nqubits} qubits"
    ):
        # gen unopt circuit
        input_seq = PauliRotationSequence(nqubits)
        initial_pauli_string = "Z" * nqubits
        input_seq.add_gate((0,), f"+{initial_pauli_string}")
        # duplicate the circuit
        # Perform unoptimization
        unopt_seq = unoptimize_circuit(
            input_seq, unopt_iteration_count, with_swap_option
        )
        if unopt_seq == "Nothing":
            continue
        with open(f"unopt_{nqubits}.pickle", "wb") as f:
            pickle.dump(unopt_seq.get_all(), f)
        # with open(f"unopt_{nqubits}.pickle", "rb") as f:
        #     seq = pickle.load(f)
        seq = unopt_seq.get_all()

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

        # st = time()
        clifford_lst, optimized_data = full_optimization(
            data, max_iter=10, show_opt_log=False
        )

        assert equiv([[], data], [clifford_lst, optimized_data]), "Optimization failed"

        if len(optimized_data) > 0:
            print("found!!!", len(optimized_data), "gates")
            break


if __name__ == "__main__":
    main()
