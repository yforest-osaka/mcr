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
    tcount_lst, time_lst = [], []
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
        start = time()
        clifford_lst, non_clifford_lst = full_optimization(circuit, show_opt_log=False)
        output_opt_qasm_file(clifford_lst, non_clifford_lst, nqubits, tmp_filepath)
        end = time()
        tcount_lst.append(len(non_clifford_lst))
        time_lst.append(end - start)
        os.remove(tmp_filepath)
    return tcount_lst, time_lst


def main():
    nqubit_lst = [i for i in range(2, 10)]
    swap_option = True

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
    print('Finished benchmarking and saved results to CSV files.')


if __name__ == "__main__":
    main()
