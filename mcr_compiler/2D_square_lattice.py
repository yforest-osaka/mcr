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


def gen_connectivity(nqubits):
    assert int(np.sqrt(nqubits)) ** 2 == nqubits, "nqubits must be a perfect square"
    edge_length = int(np.sqrt(nqubits))
    # 縦の2qubit gate
    blue, red, green, pink = [], [], [], []
    count = 0
    for i in range(0, nqubits - edge_length):
        if i != 0 and edge_length % 2 == 0 and i % edge_length == 0:
            count += 1

        if count % 2 == 0:
            blue.append((i, i + edge_length))
            count += 1
        else:
            red.append((i, i + edge_length))
            count += 1

    # 横の2qubit gate
    count = 0
    for i in range(0, nqubits - 1):
        skip_index = edge_length - 1
        if i % edge_length == skip_index:
            if edge_length % 2 == 1:
                count += 1
        elif count % 2 == 0:
            green.append((i, i + 1))
            count += 1
        else:
            pink.append((i, i + 1))
            count += 1
    return red, blue, green, pink


def process(nqubits, iteration):
    rot = PauliRotationSequence(nqubits)
    counter = 0

    interact1, interact2, interact3, interact4 = gen_connectivity(nqubits)
    for i in range(iteration):
        ## For 2D lattice Hamiltonian
        for j0, j1 in interact1 + interact2 + interact3 + interact4:
            pauli_string_lst_x = [
                "X" if k == j0 or k == j1 else "I" for k in range(nqubits)
            ]
            rot.add_gate((counter), "+" + "".join(pauli_string_lst_x))
            counter += 1
            pauli_string_lst_y = [
                "Y" if k == j0 or k == j1 else "I" for k in range(nqubits)
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
        circuit, show_opt_log=False, max_iter=nqubits
    )
    print(
        f"Optimized circuit has {len(non_clifford_lst)} non-Clifford gates. Time: {time() - st:.5f} seconds"
    )
    if nqubits == 2:
        optimal_count = 0
    else:
        optimal_count = 2 * (nqubits - 1) * iteration
    # print("is optimal?: ", optimal_count == len(non_clifford_lst))
    return optimal_count == len(non_clifford_lst)


def main():
    nqubits_lst = [i**2 for i in range(2, 6)]  # 2x2, 3x3, 4x4, 5x5
    for nqubits in nqubits_lst:
        print("=" * 30)
        print(f"nqubits = {nqubits}")
        # iteration = nqubits
        if nqubits % 2 == 1:  # For 2D lattice
            iteration = nqubits + 1
        else:
            iteration = nqubits
        val = process(nqubits, iteration)
        # if not val:
        #     print("not optimal")
        #     break
    print("finished!")


if __name__ == "__main__":
    main()
