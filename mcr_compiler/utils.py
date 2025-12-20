from typing import List, Tuple
from uuid import uuid4
import os
from copy import deepcopy
from mcr.filesave import qasm_to_pyzx
from mcr.optimize import optimize_process_pyzx
from opt_using_mcr import grouping, mcr_swap, loop_optimization
from optimizer import attempt_mcr_retry
import re


def clifford_lst_to_qasm(clifford_lst: List[Tuple[str, Tuple[int]]], filepath: str):
    data_for_qasm = []
    max_qubit_index = -1
    for gate in clifford_lst:
        qubits = gate[1]
        max_qubit_index = max(max_qubit_index, max(qubits))
        if gate[0] == "H":
            data_for_qasm.append(f"h q[{qubits[0]}];")
        elif gate[0] == "S":
            data_for_qasm.append(f"s q[{qubits[0]}];")
        elif gate[0] == "Sdg":
            data_for_qasm.append(f"sdg q[{qubits[0]}];")
        elif gate[0] == "X":
            data_for_qasm.append(f"x q[{qubits[0]}];")
        elif gate[0] == "Y":
            data_for_qasm.append(f"y q[{qubits[0]}];")
        elif gate[0] == "Z":
            data_for_qasm.append(f"z q[{qubits[0]}];")
        elif gate[0] == "CNOT":
            data_for_qasm.append(f"cx q[{qubits[0]}],q[{qubits[1]}];")
        elif gate[0] == "CZ":
            data_for_qasm.append(f"cz q[{qubits[0]}],q[{qubits[1]}];")
        elif gate[0] == "SWAP":
            data_for_qasm.append(f"swap q[{qubits[0]}],q[{qubits[1]}];")
        else:
            raise ValueError(f"Unknown gate type: {gate}")
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{max_qubit_index + 1}];",
    ]
    with open(filepath, "w") as f:
        f.write("\n".join(header + data_for_qasm))


def clifford_compile(
    clifford_lst: List[Tuple[str, Tuple[int]]], with_file_save=True
) -> str:
    unique_id = str(uuid4())
    clifford_lst_to_qasm(clifford_lst, f"tmp/clifford_{unique_id}.qasm")
    pyzx_circuit = qasm_to_pyzx(f"tmp/clifford_{unique_id}.qasm")
    os.remove(f"tmp/clifford_{unique_id}.qasm")
    pyzx_circuit_opt = optimize_process_pyzx(pyzx_circuit)
    if with_file_save:
        with open(f"tmp/clifford_opt_{unique_id}.qasm", "w") as f:
            f.write(pyzx_circuit_opt.to_qasm())
    return pyzx_circuit_opt.to_qasm()


def sub_compile(data, trial_count, reverse_input=False):
    clifford_lst = []
    current_gate_count = len(data)
    counter = 0
    for i in range(trial_count):
        tmp1 = mcr_swap(grouping(data, reverse_input=reverse_input))
        clifford, non_clifford = loop_optimization(tmp1, show_log=False)
        clifford_lst.extend(clifford)
        if len(non_clifford) == 1:
            break
        if len(non_clifford) >= current_gate_count:
            counter += 1
        if counter == 5:
            non_clifford = sum(
                attempt_mcr_retry(non_clifford, reverse_input=reverse_input), []
            )
            counter = 0
        data = deepcopy(non_clifford)
        current_gate_count = len(data)
    if len(non_clifford) == 1:
        return clifford_lst, non_clifford
    return clifford_lst, non_clifford


def compile(data_pauli_bit_lst, start_from="left"):
    data = data_pauli_bit_lst
    trial = 10
    switching_iteration = 20
    clifford_lst = []
    for iteration in range(switching_iteration):
        clifford, non_clifford = sub_compile(
            data, trial_count=trial, reverse_input=(start_from == "left")
        )
        clifford_lst.extend(clifford)
        if len(non_clifford) == 1:
            return clifford_lst, non_clifford
        data = deepcopy(non_clifford)
        clifford, non_clifford = sub_compile(
            non_clifford, trial_count=trial, reverse_input=(start_from == "right")
        )
        clifford_lst.extend(clifford)
        if len(non_clifford) == 1:
            return clifford_lst, non_clifford
        data = deepcopy(non_clifford)
    return clifford_lst, non_clifford


def extract_qasm(qasm: str) -> list[str]:
    parts = qasm.split(";")
    result = []
    found_qreg = False
    for part in parts:
        part = part.strip()
        if not part:
            continue
        stmt = part + ";"
        if found_qreg:
            result.append(stmt)
        elif part.startswith("qreg"):
            found_qreg = True
    return result


def output_opt_qasm_file(clifford_lst, non_clifford, nqubits, filepath):
    # Optimization for clifford part (using PyZX)
    header = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{nqubits}];"]
    if len(clifford_lst) > 0:
        qasm_opt_circuit = clifford_compile(clifford_lst, with_file_save=False)
        qasm_lines = header + extract_qasm(qasm_opt_circuit)
    else:
        qasm_lines = header

    # attach non-clifford gate at the end
    for elem in non_clifford:
        for gate in elem.convert_into_qasm_str():
            qasm_lines.append(f"{gate}")
    with open(filepath, "w") as f:
        f.write("\n".join(qasm_lines))
