# Clifford + T 非最適化でLitinski compileをやってみる
import os
import sys
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from pytket.circuit import OpType
from pytket.passes import FullPeepholeOptimise, RemoveRedundancies
from pytket.qasm import circuit_from_qasm
from qulacsvis import circuit_drawer as draw
from tqdm import tqdm

from mcr.circuit_ops import (
    are_unitaries_equivalent,
    equivalence_check_via_mqt_qcec,
    get_merged_matrix,
    parametric_myqc_to_rot_ops,
    unoptimization,
)
from mcr.filesave import *

# from unoptimization.litinski_compile import (
#     extract_clifford_circuit,
#     get_t_layer_from_clifford_t_circuit,
#     optimize_until_convergence,
# )
from mcr.mycircuit import MyQuantumProgram, get_gate_info
from mcr.pyzxfunc import *
from mcr.rot_class import RotOps
from mcr.rotation_circuit import (
    PauliRotationSequence,
    process_gate_replacement,
    rot_unoptimization,
)
from mcr.stim_function import (
    rot_ops_to_stim,
    zhang_optimization,
    zhang_optimization_until_convergence,
)


def get_tcount_from_tket_circuit(arg_circuit):
    return arg_circuit.n_gates_of_type(OpType.T) + arg_circuit.n_gates_of_type(OpType.Tdg)


def tket_optimization(qasm_filepath_u, qasm_filepath_v):
    # Tketの最適化(RemoveRedundancies)

    circ_tk_u = circuit_from_qasm(qasm_filepath_u)
    circ_tk_v = circuit_from_qasm(qasm_filepath_v)

    tk_gate_set = {OpType.T, OpType.Tdg, OpType.CX, OpType.H, OpType.S, OpType.Sdg, OpType.Rz}
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
    # print("tketで最適化した結果")
    # print(f"非最適化前の回路: {tket_before_tcount} -> {tket_before_opt_tcount}")
    # print(f"非最適化後の回路: {tket_after_tcount} -> {tket_after_opt_tcount}")
    return [tket_before_tcount, tket_before_opt_tcount], [tket_after_tcount, tket_after_opt_tcount]


def pyzx_optimization(qulacs_cl_t_circ_u, qulacs_cl_t_circ_v):
    pyzx_before = qulacs_to_pyzx(qulacs_cl_t_circ_u)
    pyzx_after = qulacs_to_pyzx(qulacs_cl_t_circ_v)
    pyzx_before_opt = optimize(pyzx_before)
    pyzx_after_opt = optimize(pyzx_after)

    # print("full_reduceで最適化した結果")
    # print(f"非最適化前の回路: {pyzx_before.tcount()} -> {pyzx_before_opt.tcount()}")
    # print(f"非最適化後の回路: {pyzx_after.tcount()} -> {pyzx_after_opt.tcount()}")
    return [pyzx_before.tcount(), pyzx_before_opt.tcount()], [pyzx_after.tcount(), pyzx_after_opt.tcount()]


def litinski_optimization(nqubits, stim_data_lst_u, stim_data_lst_v):
    u_optimized_rotations, u_clifford_circuit, u_process = zhang_optimization_until_convergence(
        nqubits, stim_data_lst_u, with_grouping_t_layers=True, with_process=True
    )
    v_optimized_rotations, v_clifford_circuit, v_process = zhang_optimization_until_convergence(
        nqubits, stim_data_lst_v, with_grouping_t_layers=True, with_process=True
    )

    # print("Zhangのアルゴリズムで最適化した結果")
    # print(f"非最適化前の回路: {len(stim_data_lst_u)} -> {len(u_optimized_rotations)}")
    # print(f"非最適化後の回路: {len(stim_data_lst_v)} -> {len(v_optimized_rotations)}")
    return [len(stim_data_lst_u), len(u_optimized_rotations)], [len(stim_data_lst_v), len(v_optimized_rotations)]


def get_benchmark_result(
    sample_idx,
    data_folder_path,
    nqubits,
    unopt_iteration,
    circuit_type,
    angles,
    index_select_option,
    swap_option,
    allow_3q_gate,
    run_kak,
    insertion_gate_count,
    insert_nontrivial_clifford,
    with_swap_option,
):
    qasm_folder_path = f"{data_folder_path}/n={nqubits}_k={unopt_iteration}/qasm/"
    qc_folder_path = f"{data_folder_path}/n={nqubits}_k={unopt_iteration}/qc/"

    ### 非自明Identityを挿入するだけのunoptimization ###
    # circuit = MyQuantumProgram(nqubits)
    # circuit.add_qc_gateset(arg_depth=nqubits, arg_circuit_type=circuit_type, arg_angles=angles)

    # circuit_u, circuit_v = unoptimization(
    #     arg_original_circuit=circuit,
    #     arg_unopt_iteration_count=unopt_iteration,
    #     arg_circuit_type=circuit_type,
    #     arg_index_select_option=index_select_option,
    #     arg_swap_option=swap_option,
    #     arg_allow_3q_gate=allow_3q_gate,
    #     arg_run_kak=run_kak,
    #     arg_gate_count=insertion_gate_count,
    #     arg_insert_nontrivial_clifford=insert_nontrivial_clifford,
    # )

    # u = parametric_myqc_to_rot_ops(circuit_u).convert_to_clifford_t_circuit()
    # v = parametric_myqc_to_rot_ops(circuit_v).convert_to_clifford_t_circuit()
    # circuit_u_rots = parametric_myqc_to_rot_ops(circuit_u)
    # circuit_v_rots = parametric_myqc_to_rot_ops(circuit_v)
    # if nqubits <= 6:
    #     # mqt.qcecに通すにはparametric cirucitをclifford t circuitに変換する必要がある
    #     assert equivalence_check_via_mqt_qcec(
    #         u, v, exclude_zx_checker=True, show_log=False
    #     ), "The circuit is not equivalent to the original one."
    # # else:
    # #     print("skipped due to a large number of qubits")
    #########################################

    ### stim-rotation baseのunoptimization ###
    circuit = PauliRotationSequence(nqubits)
    initial_pauli_string = "Z" * nqubits
    circuit.add_gate((0,), stim.PauliString(f"+{initial_pauli_string}"))
    # circuit.add_gate((1,), stim.PauliString(f"-{initial_pauli_string}"))
    circuit_initial = circuit.duplicate()

    # Perform unoptimization
    unopt_circuit = rot_unoptimization(circuit, unopt_iteration, with_swap_option)
    if unopt_circuit == "Nothing":
        return False
    u = circuit_initial.set_circuit()
    v = unopt_circuit.set_circuit()
    circuit_u_rots = circuit_initial.to_rot_ops()
    circuit_v_rots = unopt_circuit.to_rot_ops()
    if nqubits <= 6:
        # mqt.qcecに通すにはparametric cirucitをclifford t circuitに変換する必要がある
        assert unopt_circuit.is_equivalent(circuit_initial), "The circuit is not equivalent to the original one."
    # else:
    #     print("skipped due to a large number of qubits")
    #########################################

    if insert_nontrivial_clifford:
        filepath_u = f"{qasm_folder_path}/u_nontrivial_{sample_idx}.qasm"
        filepath_v = f"{qasm_folder_path}/v_nontrivial_{sample_idx}.qasm"
        filepath_u_qc = f"{qc_folder_path}/u_nontrivial_n_{nqubits}_k_{unopt_iteration}_{sample_idx}.qc"
        filepath_v_qc = f"{qc_folder_path}/v_nontrivial_n_{nqubits}_k_{unopt_iteration}_{sample_idx}.qc"
    else:
        filepath_u = f"{qasm_folder_path}/u_{sample_idx}_{uuid4()}.qasm"
        filepath_v = f"{qasm_folder_path}/v_{sample_idx}_{uuid4()}.qasm"
        filepath_u_qc = f"{qc_folder_path}/u_n_{nqubits}_k_{unopt_iteration}_{sample_idx}.qc"
        filepath_v_qc = f"{qc_folder_path}/v_n_{nqubits}_k_{unopt_iteration}_{sample_idx}.qc"

    qulacs_to_qasm(filepath_u, u)
    qulacs_to_qasm(filepath_v, v)
    # qasm_file_to_qc(filepath_u, filepath_u_qc)
    qasm_file_to_qc(filepath_v, filepath_v_qc)

    # 回路のフォーマット変換
    qulacs_before = circuit_u_rots.convert_to_clifford_t_circuit()
    qulacs_after = circuit_v_rots.convert_to_clifford_t_circuit()
    stim_data_lst_u = rot_ops_to_stim(circuit_u_rots)
    stim_data_lst_v = rot_ops_to_stim(circuit_v_rots)

    # Tketの最適化(RemoveRedundancies)
    tket_u_info, tket_v_info = tket_optimization(filepath_u, filepath_v)

    # PyZXの最適化(FullReduce)
    pyzx_u_info, pyzx_v_info = pyzx_optimization(qulacs_before, qulacs_after)

    # Litinskiコンパイル(Zhangのアルゴリズム)
    litinski_u_info, litinski_v_info = litinski_optimization(nqubits, stim_data_lst_u, stim_data_lst_v)

    # os.remove(filepath_u)
    # os.remove(filepath_v)
    # os.remove("u.qc")
    # os.remove("v.qc")
    result_dict = {
        "tket_u_info": tket_u_info,
        "tket_v_info": tket_v_info,
        "pyzx_u_info": pyzx_u_info,
        "pyzx_v_info": pyzx_v_info,
        "litinski_u_info": litinski_u_info,
        "litinski_v_info": litinski_v_info,
    }
    # 最後に回路データは削除
    os.remove(filepath_u)
    os.remove(filepath_v)
    return result_dict


def analyze_benchmark_result(dict_result_lst):
    data_all = []
    for data in dict_result_lst:
        ##### Tカウントの比でデータを取りたいときに使う#####
        t_original = data["litinski_u_info"][1]
        data_normalized = {key: [value / t_original for value in values] for key, values in data.items()}
        data = data_normalized
        #########################################
        data_all.append(data)

    flattened_data = []
    for entry in data_all:
        for key, values in entry.items():
            flattened_data.append({"key": key, "before": values[0], "after": values[1]})

    # Create a DataFrame
    df = pd.DataFrame(flattened_data)

    # Calculate mean and standard deviation for each key
    result = df.groupby("key").agg({"before": ["mean", "std"], "after": ["mean", "std"]}).reset_index()
    result.columns = ["key", "before_mean", "before_std", "after_mean", "after_std"]
    return result


# 並列化用の関数を定義
def process_sample(
    sample_idx,
    data_folder_path,
    nqubit,
    unopt_iteration,
    circuit_type,
    angles,
    index_select_option,
    swap_option,
    allow_3q_gate,
    run_kak,
    insertion_gate_count,
    insert_nontrivial_clifford,
    with_swap_option,
):
    return get_benchmark_result(
        sample_idx,
        data_folder_path,
        nqubit,
        unopt_iteration,
        circuit_type,
        angles,
        index_select_option,
        swap_option,
        allow_3q_gate,
        run_kak,
        insertion_gate_count,
        insert_nontrivial_clifford,
        with_swap_option,
    )


def main():
    ### Settings ###
    samples = 30  # vandaele以外のコンパイラは、100で実行
    nqubits = [i for i in range(2, 6)]  # 2, 3, 4, 5
    # nqubits = [6]
    insert_nontrivial_clifford = False
    with_swap_option = True  # for stim-rotation based unoptimization
    unopt_iterations = [nqubit**2 for nqubit in nqubits]  # 非最適化の繰り返し回数
    circuit_type = "Clifford_t"  # 'RandomUnitary' or 'Clifford_t'
    angles = [-np.pi / 4, np.pi / 4]  # only available for 'Clifford_t'
    index_select_option = "equally"  # 'equally' or 'random'
    swap_option = "keep_left"  # 'keep_left'(左側の行列成分を保持したまま右へ移動する) or 'keep_right'(右側の行列成分を保持したまま右へ移動する) or 'random'
    allow_3q_gate = False
    run_kak = False  # Trueにするとkak分解まで実行できますが速度は約1/2になります
    insertion_gate_count = 4
    data_folder_path = "results"

    #################

    for nqubit, unopt_iteration in tqdm(
        zip(nqubits, unopt_iterations), total=len(nqubits), desc="nqubit loop", leave=False
    ):
        # フォルダの作成
        qasm_folder_path = f"{data_folder_path}/n={nqubit}_k={unopt_iteration}/qasm"
        qc_folder_path = f"{data_folder_path}/n={nqubit}_k={unopt_iteration}/qc"
        if not os.path.exists(qasm_folder_path):
            os.makedirs(qasm_folder_path)
        if not os.path.exists(qc_folder_path):
            os.makedirs(qc_folder_path)

        # result_dict_lst = []
        # for sample_idx in tqdm(range(samples), total=samples, desc="sample loop", leave=False):
        #     result_dict = get_benchmark_result(
        #         sample_idx,
        #         data_folder_path,
        #         nqubit,
        #         unopt_iteration,
        #         circuit_type,
        #         angles,
        #         index_select_option,
        #         swap_option,
        #         allow_3q_gate,
        #         run_kak,
        #         insertion_gate_count,
        #         insert_nontrivial_clifford,
        #     )
        #     result_dict_lst.append(result_dict)

        # 並列処理の実行
        with parallel_backend("multiprocessing"):
            result_dict_lst = Parallel(n_jobs=-1)(
                delayed(process_sample)(
                    sample_idx,
                    data_folder_path,
                    nqubit,
                    unopt_iteration,
                    circuit_type,
                    angles,
                    index_select_option,
                    swap_option,
                    allow_3q_gate,
                    run_kak,
                    insertion_gate_count,
                    insert_nontrivial_clifford,
                    with_swap_option,
                )
                for sample_idx in tqdm(range(samples), total=samples, desc="sample loop", leave=False)
            )
        # ここでファイルの保存をしているが、保存はせずにmeanやstdを計算しておいて、1個のファイルとして保存することを考える
        # if insert_nontrivial_clifford:
        #     filename = (
        #         f"{data_folder_path}/n={nqubit}_k={unopt_iteration}/nontrivial_benchmark_result_{sample_idx}.pickle"
        #     )
        # else:
        #     filename = f"{data_folder_path}/n={nqubit}_k={unopt_iteration}/benchmark_result_{sample_idx}.pickle"
        # save_by_pickle(filename, result_dict)

        if insert_nontrivial_clifford:
            filename = f"{data_folder_path}/n={nqubit}_k={unopt_iteration}/nontrivial_benchmark_result.pickle"
        else:
            filename = f"{data_folder_path}/n={nqubit}_k={unopt_iteration}/benchmark_result.pickle"
        count = len(result_dict_lst)
        result_dict_lst = [result_dict for result_dict in result_dict_lst if result_dict]
        # print("diff: ", count - len(result_dict_lst))
        save_by_pickle(filename, analyze_benchmark_result(result_dict_lst))
    print("finished!")


if __name__ == "__main__":
    main()
