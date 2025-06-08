# 新しく作成
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyzx as zx
from pytket.circuit import OpType
from pytket.passes import RemoveRedundancies
from pytket.qasm import circuit_from_qasm

from mcr.filesave import mkdir_tmp_if_not_exists, qasm_file_to_qc, qasm_to_pyzx
from mcr.tmerge_function import zhang_optimization_until_convergence


def result_lists_to_dataframe(*result_list):
    """Function to convert results into a DataFrame

    Args:
        result_list (list): List of results

    Returns:
        pd.DataFrame: DataFrame containing the results
    """
    data = list(result_list)
    df = pd.DataFrame(data)
    df.columns = ["before_opt", "after_opt"]
    df.index = ["input_circuit", "unopted_circuit"]
    return df.T


def get_tcount_from_tket_circuit(arg_circuit):
    return arg_circuit.n_gates_of_type(OpType.T) + arg_circuit.n_gates_of_type(
        OpType.Tdg
    )


def tket_optimization(qasm_filepath_u, qasm_filepath_v):
    # Optimization using Tket (RemoveRedundancies)

    circ_tk_u = circuit_from_qasm(qasm_filepath_u)
    circ_tk_v = circuit_from_qasm(qasm_filepath_v)

    circ_tk_u_opt = circ_tk_u.copy()
    circ_tk_v_opt = circ_tk_v.copy()
    RemoveRedundancies().apply(circ_tk_u_opt)
    RemoveRedundancies().apply(circ_tk_v_opt)
    assert circ_tk_u_opt.n_gates_of_type(OpType.Rz) == 0, "Rz gate generated!!"
    assert circ_tk_v_opt.n_gates_of_type(OpType.Rz) == 0, "Rz gate generated!!"

    tket_before_tcount = get_tcount_from_tket_circuit(circ_tk_u)
    tket_after_tcount = get_tcount_from_tket_circuit(circ_tk_v)

    tket_before_opt_tcount = get_tcount_from_tket_circuit(circ_tk_u_opt)
    tket_after_opt_tcount = get_tcount_from_tket_circuit(circ_tk_v_opt)
    # print("Results of optimization using Tket")
    # print(f"Unoptimized circuit: {tket_before_tcount} -> {tket_before_opt_tcount}")
    # print(f"Optimized circuit: {tket_after_tcount} -> {tket_after_opt_tcount}")
    return result_lists_to_dataframe(
        [tket_before_tcount, tket_before_opt_tcount],
        [tket_after_tcount, tket_after_opt_tcount],
    )


def optimize_process_pyzx(
    pyzx_circuit: zx.circuit.Circuit, quiet: bool = True
) -> zx.circuit.Circuit:
    """Optimization using PyZX (max)

    Args:
        pyzx_circuit (Circuit): PyZX Circuit
        quiet (bool, optional): Whether to suppress logs. Defaults to True.

    Returns:
        Circuit: Optimized Circuit
    """
    g = pyzx_circuit.to_graph()
    zx.full_reduce(
        g, quiet=quiet
    )  # Simplifies the graph in-place and shows the rewrite steps taken.
    g.normalize()  # Makes the graph more suitable for displaying
    c_opt = zx.extract_circuit(g.copy())
    return c_opt


def pyzx_optimization(qasm_filepath_u, qasm_filepath_v):
    pyzx_before = qasm_to_pyzx(qasm_filepath_u)
    pyzx_after = qasm_to_pyzx(qasm_filepath_v)
    pyzx_before_opt = optimize_process_pyzx(pyzx_before)
    pyzx_after_opt = optimize_process_pyzx(pyzx_after)

    # print(f"Unoptimized circuit: {pyzx_before.tcount()} -> {pyzx_before_opt.tcount()}")
    # print(f"Optimized circuit: {pyzx_after.tcount()} -> {pyzx_after_opt.tcount()}")
    return result_lists_to_dataframe(
        [pyzx_before.tcount(), pyzx_before_opt.tcount()],
        [pyzx_after.tcount(), pyzx_after_opt.tcount()],
    )


def tmerge_optimization(nqubits, input_seq, unopted_seq):
    input_seq_stim = input_seq.sort_gate_sequence(only_gates=True)
    unopted_seq_stim = unopted_seq.sort_gate_sequence(only_gates=True)

    input_qc_optimized_rotations, _, _ = zhang_optimization_until_convergence(
        nqubits, input_seq_stim, with_grouping_t_layers=True, with_process=True
    )
    unopted_qc_optimized_rotations, _, _ = zhang_optimization_until_convergence(
        nqubits, unopted_seq_stim, with_grouping_t_layers=True, with_process=True
    )

    # print("Results of optimization using Zhang's algorithm")
    # print(f"Unoptimized circuit: {len(stim_data_lst_u)} -> {len(input_qc_optimized_rotations)}")
    # print(f"Optimized circuit: {len(stim_data_lst_v)} -> {len(unopted_qc_optimized_rotations)}")
    return result_lists_to_dataframe(
        [len(input_seq_stim), len(input_qc_optimized_rotations)],
        [len(unopted_seq_stim), len(unopted_qc_optimized_rotations)],
    )


def analyze_qc_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- Process the first line (ancilla count) ---
    header = lines[0].strip().split()
    filtered = [item for item in header if not (item == ".v" or item.startswith("q"))]
    ancilla_count = len(filtered)

    # --- Count "T" from the second line onwards ---
    T_count = 0
    for line in lines[1:]:
        parts = line.strip().split()
        if parts and parts[0] == "T":
            T_count += 1

    return ancilla_count, T_count


def fasttodd_optimization(unopted_circuit_filepath):
    mkdir_tmp_if_not_exists()
    tmp_qc_filepath = f"tmp/{uuid4()}.qc"
    abs_tmp_qc_filepath = str(Path(tmp_qc_filepath).resolve())
    qasm_file_to_qc(unopted_circuit_filepath, tmp_qc_filepath)
    current_dir = os.path.basename(os.getcwd())
    filename = Path(tmp_qc_filepath).name

    if current_dir == "script":
        work_dir = "../quantum-circuit-optimization"
    else:
        work_dir = "./quantum-circuit-optimization"

    # Define the output file path beforehand
    output_filepath = str(Path(f"{work_dir}/circuits/outputs/{filename}").resolve())
    os.chdir(work_dir)
    process = subprocess.Popen(
        ["cargo", "+nightly", "run", "-r", abs_tmp_qc_filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()
    os.chdir("..")  # Change back to the original directory

    if os.path.exists(output_filepath):
        ancilla, T = analyze_qc_file(output_filepath)
        os.remove(output_filepath)
        os.remove(tmp_qc_filepath)
        df = pd.DataFrame({"added_ancilla": [ancilla], "after_opt_T_count": [T]})
        return df
    else:
        return {"error": "Output file was not found"}
