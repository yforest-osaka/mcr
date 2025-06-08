import os
from mcr.optimize import (
    fasttodd_optimization,
    pyzx_optimization,
    tket_optimization,
    tmerge_optimization,
)
from mcr.rotation_circuit import PauliRotationSequence
from mcr.unoptimize import unoptimize_circuit
from joblib import Parallel, delayed
from time import time
from uuid import uuid4
from tqdm import tqdm
import pandas as pd
import sys


def tket_process(qasm_filepath_u, qasm_filepath_v):
    st = time()
    df = tket_optimization(qasm_filepath_u, qasm_filepath_v)
    ed = time()
    t_before = df["unopted_circuit"]["before_opt"]
    t_aft = df["unopted_circuit"]["after_opt"]
    return t_before, t_aft, ed - st


def pyzx_process(qasm_filepath_u, qasm_filepath_v):
    st = time()
    df = pyzx_optimization(qasm_filepath_u, qasm_filepath_v)
    ed = time()
    # t_before = df["unopted_circuit"]["before_opt"]
    t_aft = df["unopted_circuit"]["after_opt"]
    return t_aft, ed - st


def tmerge_process(nqubits, input_seq, unopt_seq):
    st = time()
    df = tmerge_optimization(nqubits, input_seq, unopt_seq)
    ed = time()
    # t_before = df["unopted_circuit"]["before_opt"]
    t_aft = df["unopted_circuit"]["after_opt"]
    return t_aft, ed - st


def fasttodd_process(qasm_filepath_v):
    st = time()
    df = fasttodd_optimization(qasm_filepath_v)
    ed = time()
    ancilla_count = df["added_ancilla"][0]
    t_aft = df["after_opt_T_count"][0]
    return t_aft, ancilla_count, ed - st


def main():
    ### Input information (parameters) ###
    num_samples = 30
    nqubits = int(sys.argv[1])  # Number of qubits, e.g., 3, 4, 5, etc.
    cpu_count = -1  # Number of CPU cores to use for the optimization
    # nqubits = 3

    with_swap_option = False  # If True, the MCR swap is executed (then the unoptimized circuit becomes longer)
    if with_swap_option:
        folder_name = "results_with_swap"
    else:
        folder_name = "results"
    # Number of iterations for the unoptimized circuit
    unopt_iteration_count = nqubits**2
    ############################
    input_circuit_data = []
    output_circuit_data = []
    input_circuit_filepath_lst = []
    output_circuit_filepath_lst = []

    for _ in range(num_samples):
        input_seq = PauliRotationSequence(nqubits)
        initial_pauli_string = "Z" * nqubits
        input_seq.add_gate((0,), f"+{initial_pauli_string}")
        # duplicate the circuit
        initial_seq = input_seq.duplicate()

        # Perform unoptimization
        unopt_seq = unoptimize_circuit(
            input_seq, unopt_iteration_count, with_swap_option
        )
        if nqubits <= 4:
            assert unopt_seq.is_equivalent(initial_seq), (
                "The circuit is not equivalent to the original one."
            )

        # Save input and output circuits by QASM format
        circ_id = uuid4()
        input_circuit_filepath = f"/tmp/u_{circ_id}.qasm"
        unopted_circuit_filepath = f"/tmp/v_{circ_id}.qasm"

        initial_seq.save_qasm(input_circuit_filepath)
        unopt_seq.save_qasm(unopted_circuit_filepath)
        input_circuit_data.append(initial_seq)
        output_circuit_data.append(unopt_seq)
        input_circuit_filepath_lst.append(input_circuit_filepath)
        output_circuit_filepath_lst.append(unopted_circuit_filepath)

    # Compiler evaluation
    # Pytket optimization
    result_tket = Parallel(n_jobs=cpu_count)(
        delayed(tket_process)(input_circuit_filepath, unopted_circuit_filepath)
        for input_circuit_filepath, unopted_circuit_filepath in tqdm(
            zip(input_circuit_filepath_lst, output_circuit_filepath_lst),
            total=num_samples,
            desc="Processing circuits with Tket",
        )
    )
    tket_tcount_bef, tket_tcount_aft, tket_time = zip(*result_tket)
    df = pd.DataFrame(
        {
            "before_T_count": tket_tcount_bef,
            "tket_after_T_count": tket_tcount_aft,
            "tket_time": tket_time,
        }
    )
    df.to_csv(f"{folder_name}/n={nqubits}_k={nqubits**2}.csv", index=False)

    # PyZX optimization
    result_pyzx = Parallel(n_jobs=cpu_count)(
        delayed(pyzx_process)(input_circuit_filepath, unopted_circuit_filepath)
        for input_circuit_filepath, unopted_circuit_filepath in tqdm(
            zip(input_circuit_filepath_lst, output_circuit_filepath_lst),
            total=num_samples,
            desc="Processing circuits with PyZX",
        )
    )
    pyzx_tcount_aft, pyzx_time = zip(*result_pyzx)
    df["pyzx_after_T_count"] = pyzx_tcount_aft
    df["pyzx_time"] = pyzx_time
    df.to_csv(f"{folder_name}/n={nqubits}_k={nqubits**2}.csv", index=False)

    # TMerge optimization
    result_tmerge = Parallel(n_jobs=cpu_count)(
        delayed(tmerge_process)(nqubits, input_seq, unopt_seq)
        for input_seq, unopt_seq in tqdm(
            zip(input_circuit_data, output_circuit_data),
            total=num_samples,
            desc="Processing circuits with TMerge",
        )
    )
    tmerge_tcount_aft, tmerge_time = zip(*result_tmerge)
    df["tmerge_after_T_count"] = tmerge_tcount_aft
    df["tmerge_time"] = tmerge_time
    df.to_csv(f"{folder_name}/n={nqubits}_k={nqubits**2}.csv", index=False)

    # FastTODD optimization
    truncation = int(num_samples * 1.0)
    result_fasttodd = Parallel(n_jobs=cpu_count)(
        delayed(fasttodd_process)(unopted_circuit_filepath)
        for unopted_circuit_filepath in tqdm(
            output_circuit_filepath_lst[:truncation],
            total=truncation,
            desc="Processing circuits with FastTODD",
        )
    )
    fasttodd_tcount_aft, ancilla_count, fasttodd_time = zip(*result_fasttodd)
    df = pd.DataFrame()
    df["fasttodd_after_T_count"] = fasttodd_tcount_aft
    df["fasttodd_ancilla_count"] = ancilla_count
    df["fasttodd_time"] = fasttodd_time
    df.to_csv(f"{folder_name}/n={nqubits}_k={nqubits**2}_fasttodd.csv", index=False)

    for elem in input_circuit_filepath_lst + output_circuit_filepath_lst:
        os.remove(elem)
    print("Finished the optimization of the circuit")


if __name__ == "__main__":
    main()
