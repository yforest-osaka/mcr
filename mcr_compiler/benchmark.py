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


def benchmark(nqubits, with_swap):
    if with_swap:
        data = Path(f"circuit_data/with_swap/{nqubits}")
    else:
        data = Path(f"circuit_data/without_swap/{nqubits}")
    tcount_lst_before, tcount_lst_after, time_lst = [], [], []
    for i, file in tqdm(
        enumerate(natsorted(data.iterdir())),
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
        tcount_lst_before.append(len(data_input))
        start = time()
        clifford_lst, non_clifford_lst = full_optimization(
            circuit, show_opt_log=False, max_iter=nqubits
        )
        output_opt_qasm_file(clifford_lst, non_clifford_lst, nqubits, tmp_filepath)
        end = time()
        tcount_lst_after.append(len(non_clifford_lst))
        time_lst.append(end - start)
        os.remove(tmp_filepath)
    return tcount_lst_before, tcount_lst_after, time_lst


def main():
    nqubit_lst = [i for i in range(2, 10)]
    swap_option = False

    tcount_result_before, tcount_result_after, time_result, reduction_rates = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    for nqubits in tqdm(nqubit_lst, desc="Benchmarking"):
        tcount_data_before, tcount_data_after, time_data = benchmark(
            nqubits, swap_option
        )
        tcount_result_before[nqubits] = tcount_data_before
        tcount_result_after[nqubits] = tcount_data_after
        time_result[nqubits] = time_data
        reduction_rates[nqubits] = [
            100 * (before - after) / (before - 1)
            for before, after in zip(tcount_data_before, tcount_data_after)
        ]

    print("-" * 30)
    print("T-count Results (Before Optimization):")
    print(tcount_result_before.describe())
    print("T-count Results (After Optimization):")
    print(tcount_result_after.describe())
    print("\nCompilation Time Results:")
    print(time_result.describe())
    print("\nT-count Reduction Rates (%):")
    print(reduction_rates.describe())
    print("-" * 30)


if __name__ == "__main__":
    main()
