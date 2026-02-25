from pathlib import Path
from natsort import natsorted
import pandas as pd
from time import time
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from mcr.gate_apply import PauliBit
from mcr.rotation_circuit import load_circuit_from_txt
from mcr.mcr_optimize import full_optimization, output_opt_qasm_file
import os
from mcr.equiv_check import equiv
from mcr.mcr_optimize import loop_optimization
from mcr.rotation_circuit import PauliRotationSequence
from time import time
from mcr.optimize import zhang_optimization_until_convergence


def process(nqubits, iteration):
    rot = PauliRotationSequence(nqubits)
    counter = 0

    for i in range(iteration):
        # For 1D chain Hamiltonian
        for j in range(0, nqubits - 1):
            # for j in range(0, nqubits):
            pauli_string_lst_x = [
                "X" if k == j or k == (j + 1) % nqubits else "I" for k in range(nqubits)
            ]
            rot.add_gate((counter), "+" + "".join(pauli_string_lst_x))
            counter += 1
            pauli_string_lst_y = [
                "Y" if k == j or k == (j + 1) % nqubits else "I" for k in range(nqubits)
            ]
            rot.add_gate((counter), "+" + "".join(pauli_string_lst_y))
            counter += 1

        for j in range(nqubits):
            pauli_string_lst = ["Z" if k == j else "I" for k in range(nqubits)]
            rot.add_gate((counter), "+" + "".join(pauli_string_lst))
            counter += 1

    rot.save_circuit_using_txt("sample.txt")

    rot = PauliRotationSequence(nqubits)
    filepath = "sample.txt"
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line == "":
                continue
            rot.add_gate((i,), line)

    # TMerge
    # print(f"Original circuit has {rot.get_gate_count()} non-Clifford gates.")
    optimized_rot, _, _ = zhang_optimization_until_convergence(
        nqubits,
        rot.sort_gate_sequence(only_gates=True),
        with_grouping_t_layers=True,
        with_process=True,
    )

    # print(f"Optimized circuit has {len(optimized_rot)} non-Clifford gates.")
    print(f"In TMerge: {rot.get_gate_count()} -> {len(optimized_rot)}")

    # My Compiler
    circuit = load_circuit_from_txt(filepath)
    data_input = []
    for elem in circuit.get_all():
        sgn = str(elem[1])[0]
        pauli_str = str(elem[1])[1:]
        if sgn == "+":
            data_input.append(PauliBit(pauli_str, np.pi / 4))
        else:
            assert sgn == "-", f"Unexpected sign: {sgn}"
            data_input.append(PauliBit(pauli_str, -np.pi / 4))
    print(f"Original circuit has {len(data_input)} non-Clifford gates.")
    st = time()
    clifford_lst, non_clifford_lst = full_optimization(
        circuit, max_iter=nqubits, show_opt_log=False
    )
    print(
        f"Optimized circuit has {len(non_clifford_lst)} non-Clifford gates. Time: {time() - st:.5f} seconds"
    )
    if nqubits == 2:
        optimal_count = 0
    else:
        optimal_count = 2 * (nqubits - 1) * iteration
    print("is optimal?: ", optimal_count == len(non_clifford_lst))
    return optimal_count == len(non_clifford_lst)


def main():
    nqubits_lst = [6 * i for i in range(1, 5)]
    for nqubits in nqubits_lst:
        print("=" * 30)
        print(f"nqubits = {nqubits}")
        iteration = nqubits
        val = process(nqubits, iteration)
        if not val:
            print("not optimal")
            break
    print("finished!")


if __name__ == "__main__":
    main()
