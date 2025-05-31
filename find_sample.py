from mcr.gate_apply import PauliBit
from mcr.mcr_optimize import find_mcr, find_nontrivial_swap
from mcr.gate_apply import grouping, loop_optimization
from mcr.equiv_check import (
    pauli_bit_equivalence_check,
    equivalence_check_via_mqt_qcec,
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
from opt_using_mcr import test_algorithm


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
        st = time()
        clifford_lst, optimized_data = test_algorithm(data, show_opt_log=False)
        ed = time()
        if len(optimized_data) > 0 and len(optimized_data) < 12:
            print("found!!!", len(optimized_data), "gates")
            break
        # print(f"Optimization time: {ed - st} seconds")
        # for elem in grouping(optimized_data):
        #     print(f"Group size: {len(elem)}")
        #     print(elem)
        #     print("==========================")

        # if nqubits <= 6:
        #     circuit_input = QuantumCircuit(nqubits)
        #     for elem in data:
        #         circuit_input.merge_circuit(elem.convert_into_qulacs())

        #     circuit_output = QuantumCircuit(nqubits)
        #     circuit_output = set_clifford_to_qulacs(circuit_output, clifford_lst)
        #     for elem in optimized_data:
        #         circuit_output.merge_circuit(elem.convert_into_qulacs())

        #     equivalence_check_via_mqt_qcec(
        #         circuit_input, circuit_output, exclude_zx_checker=True
        #     )


if __name__ == "__main__":
    main()
