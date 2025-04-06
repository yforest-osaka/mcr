import ast
import random
import re
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from qulacs import QuantumCircuit
from tqdm import tqdm

from mcr.pauli_bit_ops import multiply_pauli_bits, pauli_bit_to_pauli_id
from mcr.rot_class import RotOps, satisfies_litinski_condition


def initialize_t_gate2(data, t_position, qubit_count):
    gate_name, qubits = data[t_position]
    assert gate_name.lower() in ["t", "tdg"], f"Gate name is not T or Tdg: {gate_name}"
    gate_data = ["Z" if i == qubits[0] else "I" for i in range(qubit_count)]
    return [0 if gate_name.lower() == "t" else 2] + gate_data, set(qubits)


def get_non_identity_qubit_indices(pauli_bit):
    return [i for i, elem in enumerate(pauli_bit[1:]) if elem != "I"]  # Identity以外のindexを取得


CLIFFORD_MAP_1Q = {
    "x": {"I": "I", "X": "X", "Y": "-Y", "Z": "-Z"},
    "y": {"I": "I", "X": "-X", "Y": "Y", "Z": "-Z"},
    "z": {"I": "I", "X": "-X", "Y": "-Y", "Z": "Z"},
    "h": {"I": "I", "X": "Z", "Y": "-Y", "Z": "X"},
    "s": {"I": "I", "X": "Y", "Y": "-X", "Z": "Z"},
    "sdg": {"I": "I", "X": "-Y", "Y": "X", "Z": "Z"},
}

CLIFFORD_MAP_2Q = {
    "cz": {
        "II": "II",
        "IX": "ZX",
        "IY": "ZY",
        "IZ": "IZ",
        "XI": "XZ",
        "XX": "YY",
        "XY": "-YX",
        "XZ": "XI",
        "YI": "YZ",
        "YX": "-XY",
        "YY": "XX",
        "YZ": "YI",
        "ZI": "ZI",
        "ZX": "IX",
        "ZY": "IY",
        "ZZ": "ZZ",
    },
    "cx": {
        "II": "II",
        "IX": "IX",
        "IY": "ZY",
        "IZ": "ZZ",
        "XI": "XX",
        "XX": "XI",
        "XY": "YZ",
        "XZ": "-YY",
        "YI": "YX",
        "YX": "YI",
        "YY": "-XZ",
        "YZ": "XY",
        "ZI": "ZI",
        "ZX": "ZX",
        "ZY": "IY",
        "ZZ": "IZ",
    },
    "cx_rev": {
        "II": "II",
        "IX": "XX",
        "IY": "XY",
        "IZ": "IZ",
        "XI": "XI",
        "XX": "IX",
        "XY": "IY",
        "XZ": "XZ",
        "YI": "YZ",
        "YX": "ZY",
        "YY": "-ZX",
        "YZ": "YI",
        "ZI": "ZZ",
        "ZX": "-YY",
        "ZY": "YX",
        "ZZ": "ZI",
    },
    "swap": {
        "II": "II",
        "IX": "XI",
        "IY": "YI",
        "IZ": "ZI",
        "XI": "IX",
        "XX": "XX",
        "XY": "YX",
        "XZ": "ZX",
        "YI": "IY",
        "YX": "XY",
        "YY": "YY",
        "YZ": "ZY",
        "ZI": "IZ",
        "ZX": "XZ",
        "ZY": "YZ",
        "ZZ": "ZZ",
    },
}


def t_swap2(t_gate_pauli_bit_and_set, t_position, data_all, direction="right"):
    t_gate_pauli_bit = t_gate_pauli_bit_and_set[0].copy()
    set_qubits = t_gate_pauli_bit_and_set[1].copy()
    if direction not in ["right", "left"]:
        raise ValueError(f"Invalid direction: {direction}")
    if direction == "right":
        search_data = data_all[t_position + 1 :]
    else:
        search_data = data_all[:t_position][::-1]

    for elem in search_data:
        gate_name, qubit_lst = elem
        if gate_name.lower() in ["t", "tdg"]:
            continue  # T or Tdag gateはスキップ
        else:
            current_pauli_bit = t_gate_pauli_bit[1:]

            # current_non_idenity_qubit_indices = get_non_identity_qubit_indices(t_gate_pauli_bit)
            intersection_indices = set_qubits & set(qubit_lst)
            if not intersection_indices:  # 共通するqubitが1つもない場合→何もしなくて良い
                continue
            elif gate_name in ["x", "y", "z", "h", "s", "sdg"]:  # 1qubit Pauliの更新が必要

                tg = qubit_lst[0]  # 挟みたいclifford gateのindex
                # S, Sdagゲートで右から左へスワップしたい時はmapが少し変更される
                if gate_name == "s" and direction == "left":
                    gate_name = "sdg"
                if gate_name == "sdg" and direction == "left":
                    gate_name = "s"

                new_element = CLIFFORD_MAP_1Q[f"{gate_name}"][
                    current_pauli_bit[tg]
                ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
                # new_element = ast.literal_eval(new_element)
                if "-" in new_element:
                    t_gate_pauli_bit[0] = (t_gate_pauli_bit[0] + 2) % 4  # 符号の情報を更新
                    new_element = new_element[1:]  # 符号の情報を除去
                t_gate_pauli_bit[tg + 1] = new_element[0]  # 1qubit Pauliの更新
                if new_element[0] == "I":
                    set_qubits.remove(tg)
            else:  # 2qubit Pauliの更新が必要
                assert gate_name in ["cx", "cz", "swap"], f"Invalid gate name: {gate_name}"
                ctrl, tg = qubit_lst
                if tg - ctrl < 0 and gate_name == "cx":
                    gate_name = "cx_rev"
                    ctrl, tg = tg, ctrl
                tmp_ctrl = current_pauli_bit[ctrl]
                tmp_tg = current_pauli_bit[tg]
                new_element = CLIFFORD_MAP_2Q[f"{gate_name}"][
                    tmp_ctrl + tmp_tg
                ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
                # new_element = ast.literal_eval(new_element)
                if "-" in new_element:
                    t_gate_pauli_bit[0] = (t_gate_pauli_bit[0] + 2) % 4  # 符号の情報を更新
                    new_element = new_element[1:]  # 符号の情報を除去
                t_gate_pauli_bit[ctrl + 1] = new_element[0]
                t_gate_pauli_bit[tg + 1] = new_element[1]
                if new_element[0] == "I":
                    set_qubits.discard(ctrl)
                else:
                    set_qubits.add(ctrl)
                if new_element[1] == "I":
                    set_qubits.discard(tg)
                else:
                    set_qubits.add(tg)
    return t_gate_pauli_bit


def litinski_compile_circuit2(circuit_file, direction="right", cpu_count=1):
    st = time()
    if isinstance(circuit_file, QuantumCircuit):  # qulacsの回路が入力されたと想定
        from qulacs.converter import convert_qulacs_circuit_to_QASM

        string = convert_qulacs_circuit_to_QASM(circuit_file)
    else:  # qasmファイルが入力されたと想定
        with open(circuit_file, mode="r") as f:
            string = f.read().splitlines()
    circ_qasm = [line for line in string if line != ""]

    for ele in circ_qasm:
        if "qreg" in ele:
            qubit_count = int(re.findall(r"\d+", ele)[0])
            break

    exclude = ["OPENQASM", "include", "qreg", "creg", "measure"]
    circ_qasm = [line for line in circ_qasm if not any([word in line for word in exclude])]

    # Define a regular expression pattern to match gate names and qubit numbers
    pattern = re.compile(r"(\w+)\s+q\[(\d+)\](?:,q\[(\d+)\])?;")

    # Extract gate names and qubit numbers
    extracted_gates = []
    for line in circ_qasm:
        match = pattern.match(line)
        if match:
            gate_name = match.group(1)
            qubits = [int(match.group(2))]
            if match.group(3):
                qubits.append(int(match.group(3)))
            extracted_gates.append((gate_name, qubits))

    t_positions = []
    for i, (gate_name, qubits) in enumerate(extracted_gates):
        if gate_name.lower() == "t" or gate_name.lower() == "tdg":
            t_positions.append(i)
    print("gate_count", len(extracted_gates))
    print("t_count", len(t_positions))
    print("get t_position time:", time() - st)
    st = time()
    # results = []
    results = Parallel(n_jobs=cpu_count)(
        delayed(t_swap2)(
            initialize_t_gate2(extracted_gates, t_position, qubit_count),
            t_position,
            extracted_gates,
            direction=direction,
        )
        for t_position in tqdm(t_positions, desc="Litinski compile...", leave=False)
    )
    print("compilation time:", time() - st)
    return results


if __name__ == "__main__":
    direction = "right"
    filepath = "../data/transpiled_qasm/gf2^16_mult.qasm"
    result = litinski_compile_circuit2(filepath, direction=direction, cpu_count=10)
    from mcr.filesave import save_by_pickle

    sgns = []
    paulis = []
    for ele in result:
        if ele[0] == 2:
            sgns.append(-1)
        else:
            sgns.append(1)
        paulis.append("".join(ele[1:]))
    rot = RotOps(paulis)
    rot.insert_angles_from_sgn(sgns)
    save_by_pickle(f"sample_{direction}.pickle", rot)
