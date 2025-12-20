# load files
from pathlib import Path
import pandas as pd

# ===== Standard Library =====
from time import time
from copy import deepcopy
from typing import List, Tuple

# ===== Third-party =====
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qulacs import QuantumCircuit

# ===== Local Modules =====
from mcr.gate_apply import PauliBit, set_clifford_to_qulacs, grouping
from mcr.equiv_check import equiv, equivalence_check_via_mqt_qcec
from mcr.rotation_circuit import PauliRotationSequence, load_circuit_from_txt
from optimizer import full_optimization
import os

from mcr.filesave import qasm_to_pyzx
from mcr.optimize import optimize_process_pyzx
from utils import clifford_lst_to_qasm, compile, output_opt_qasm_file


def benchmark(nqubits, with_swap):
    if with_swap:
        data = Path(f"circuit_data/with_swap/{nqubits}")
    else:
        data = Path(f"circuit_data/without_swap/{nqubits}")
    tcount_lst, time_lst = [], []
    for i, file in tqdm(
        enumerate(data.iterdir()),
        total=len(list(data.iterdir())),
        desc="Compiling circuits",
        leave=False,
    ):
        tmp_filepath = f"circuit_{i}_opt.qasm"
        circuit = load_circuit_from_txt(file)
        data_input = []
        for elem in circuit.get_all():
            sgn = str(elem[1])[0]
            pauli_str = str(elem[1])[1:]
            if sgn == "+":
                data_input.append(PauliBit(pauli_str, np.pi / 4))
            else:
                assert sgn == "-", f"Unexpected sign: {sgn}"
                data_input.append(PauliBit(pauli_str, -np.pi / 4))
        start = time()
        # clifford_lst, non_clifford_lst = compile(data_input, start_from="left")
        clifford_lst, non_clifford_lst = full_optimization(circuit, show_opt_log=False)
        # if len(non_clifford_lst) != 1:
        #     print(len(non_clifford_lst))
        output_opt_qasm_file(clifford_lst, non_clifford_lst, nqubits, tmp_filepath)
        end = time()
        # assert equivalence_check_via_mqt_qcec(
        #     tmp_filepath,
        #     f"circuit_data/optimal/z_{nqubits}.qasm",
        #     show_log=False,
        #     exclude_zx_checker=True,
        # ), f"equivalence check failed for trial {i}"
        tcount_lst.append(len(non_clifford_lst))
        time_lst.append(end - start)
        os.remove(tmp_filepath)

    # return np.mean(tcount_lst), np.std(tcount_lst), np.mean(time_lst), np.std(time_lst)
    return tcount_lst, time_lst


def main():
    nqubit_lst = [i for i in range(2, 10)]
    swap_option = False

    tcount_result, time_result = pd.DataFrame(), pd.DataFrame()

    for nqubits in tqdm(nqubit_lst, desc="Benchmarking"):
        tcount_data, time_data = benchmark(nqubits, swap_option)
        tcount_result[nqubits] = tcount_data
        time_result[nqubits] = time_data

    print("-" * 30)
    print("T-count Results:")
    print(tcount_result.describe())
    print("\nCompilation Time Results:")
    print(time_result.describe())
    print("-" * 30)
    # Save results to CSV files
    tcount_result.to_csv(
        f"benchmark_tcount_{'with_swap' if swap_option else 'without_swap'}.csv",
        index=False,
    )
    time_result.to_csv(
        f"benchmark_time_{'with_swap' if swap_option else 'without_swap'}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
