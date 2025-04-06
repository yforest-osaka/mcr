from glob import glob
from pathlib import Path

import numpy as np
import pyzx as zx
from natsort import natsorted
from pytket import Circuit as TketCircuit
from pytket.circuit import Unitary1qBox, Unitary2qBox
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import transpile
from qiskit.qasm2 import dump, load
from qiskit.quantum_info.operators.operator import Operator
from qulacs import QuantumCircuit as QulacsCircuit
from qulacs.gate import CNOT, CZ, RX, RY, RZ, PauliRotation  # type: ignore
from tqdm import tqdm

from mcr.filesave import pyzx_to_qasm, qasm_to_pyzx, qasm_to_qulacs


def qulacs_to_qiskit(circuit_qulacs: QulacsCircuit) -> QiskitCircuit:
    """
    QulacsのQuantumCircuitをQiskitのQuantumCircuitに変換

    パラメータ:
    circuit_qulacs (QulacsCircuit): 変換するQulacsのQuantumCircuit。

    戻り値:
    QiskitCircuit: 変換されたQiskitのQuantumCircuit。

    例外:
    ValueError: 無効なゲートタイプが見つかった場合。
    """
    circ_qiskit = QiskitCircuit(circuit_qulacs.get_qubit_count())
    gate_groups = ["DenseMatrix", "Pauli-rotation", "X-rotation", "Y-rotation", "Z-rotation"]
    for i in range(circuit_qulacs.get_gate_count()):
        gate = circuit_qulacs.get_gate(i)
        gate_type = gate.get_name()
        control = gate.get_control_index_list()
        target = gate.get_target_index_list()
        if gate_type == "CNOT":
            circ_qiskit.cx(control[0], target[0])
        elif gate_type == "CZ":
            circ_qiskit.cz(control[0], target[0])
        elif gate_type == "X":
            circ_qiskit.x(target[0])
        elif gate_type == "Y":
            circ_qiskit.y(target[0])
        elif gate_type == "Z":
            circ_qiskit.z(target[0])
        elif gate_type == "H":
            circ_qiskit.h(target[0])
        elif gate_type == "S":
            circ_qiskit.s(target[0])
        elif gate_type == "T":
            circ_qiskit.t(target[0])
        elif gate_type == "Sdag":
            circ_qiskit.sdg(target[0])
        elif gate_type == "Tdag":
            circ_qiskit.tdg(target[0])
        elif gate_type in gate_groups:
            unitary = gate.get_matrix()
            circ_qiskit.append(Operator(unitary), target)
        else:
            raise ValueError(f"Invalid gate type: {gate_type}")
    return circ_qiskit


def qulacs_to_tket(circuit_qulacs: QulacsCircuit) -> TketCircuit:
    """
    QulacsのQuantumCircuitをTketのCircuitに変換

    パラメータ:
    circuit_qulacs (QulacsCircuit): 変換するQulacsのQuantumCircuit。

    戻り値:
    TketCircuit: 変換されたTketのCircuit。

    例外:
    ValueError: 無効なゲートタイプが見つかった場合。
    """
    nqubits = circuit_qulacs.get_qubit_count()
    circ_tket = TketCircuit(nqubits)
    for i in range(circuit_qulacs.get_gate_count()):
        gate = circuit_qulacs.get_gate(i)
        gate_type = gate.get_name()
        control = [(nqubits - 1) - i for i in gate.get_control_index_list()]
        control.reverse()
        target = [(nqubits - 1) - i for i in gate.get_target_index_list()]
        target.reverse()

        if gate_type == "CNOT":
            circ_tket.CX(control[0], target[0])
        elif gate_type == "CZ":
            circ_tket.CZ(control[0], target[0])
        elif gate_type == "X":
            circ_tket.X(target[0])
        elif gate_type == "Y":
            circ_tket.Y(target[0])
        elif gate_type == "Z":
            circ_tket.Z(target[0])
        elif gate_type == "H":
            circ_tket.H(target[0])
        elif gate_type == "S":
            circ_tket.S(target[0])
        elif gate_type == "T":
            circ_tket.T(target[0])
        elif gate_type in ["X-rotation", "Y-rotation", "Z-rotation"]:
            unitary = gate.get_matrix()
            circ_tket.add_gate(Unitary1qBox(unitary), target)
        elif gate_type in ["DenseMatrix", "Pauli-rotation"]:
            unitary = gate.get_matrix()
            circ_tket.add_gate(Unitary2qBox(unitary), target)
        else:
            raise ValueError(f"Invalid gate type: {gate_type}")
    return circ_tket


def cirq_gates_to_qulacs(cirq_gates: list, dem_index: list[int]) -> list:
    """
    CirqのゲートのリストをQulacsのゲートのリストに変換します。

    パラメータ:
    cirq_gates (List[cirq.Gate]): 変換するCirqのゲートのリスト。
    dem_index (List[int]): インデックスのリスト。

    戻り値:
    List[QuantumGateBase]: 変換されたQulacsのゲートのリスト。

    例外:
    ValueError: 無効なゲートタイプが見つかった場合。
    """
    qulacs_gates = []
    one_qubit_gate_mapping = {
        "PhasedXPowGate": lambda gate, index: [
            RZ(index, gate.phase_exponent * np.pi),
            RX(index, -1 * gate.exponent * np.pi),
            RZ(index, -1 * gate.phase_exponent * np.pi),
        ],
        "Rx": lambda gate, index: [RX(index, -1 * gate.exponent * np.pi)],
        "Ry": lambda gate, index: [RY(index, -1 * gate.exponent * np.pi)],
        "Rz": lambda gate, index: [RZ(index, -1 * gate.exponent * np.pi)],
    }
    two_qubit_gate_mapping = {
        "CXPowGate": lambda gate, index1, index2: [CNOT(index1, index2)],
        "CZPowGate": lambda gate, index1, index2: [CZ(index1, index2)],
        "XXPowGate": lambda gate, index1, index2: [PauliRotation([index1, index2], [1, 1], -1 * gate.exponent * np.pi)],
        "YYPowGate": lambda gate, index1, index2: [PauliRotation([index1, index2], [2, 2], -1 * gate.exponent * np.pi)],
        "ZZPowGate": lambda gate, index1, index2: [PauliRotation([index1, index2], [3, 3], -1 * gate.exponent * np.pi)],
    }
    for i in range(len(cirq_gates)):
        gate = cirq_gates[i].gate
        gate_name = type(gate).__name__
        index_list = [qubit.x for qubit in cirq_gates[i].qubits]
        index = dem_index[len(dem_index) - dem_index.index(index_list[0]) - 1]

        if len(index_list) == 1:
            if gate_name in one_qubit_gate_mapping:
                qulacs_gates.extend(one_qubit_gate_mapping[gate_name](gate, index))
            else:
                phase_z = gate.exponent * np.pi
                qulacs_gates.append(RZ(index, -1 * phase_z))
        elif len(index_list) == 2:
            index_2 = dem_index[len(dem_index) - dem_index.index(index_list[1]) - 1]
            if gate_name in two_qubit_gate_mapping:
                qulacs_gates.extend(two_qubit_gate_mapping[gate_name](gate, index, index_2))
            else:
                raise ValueError(f"Invalid gate type: {gate_name}")
        else:
            raise ValueError(f"Invalid gate type: {gate_name}")
    return qulacs_gates


def transform_ccz_to_ccx(qasm_code):
    lines = qasm_code.split("\n")
    transformed_lines = []

    for line in lines:
        if "ccz" in line:
            # Extract the qubit indices from the ccz line
            qubits = line[line.find("q[") :].split("q[")[1:]
            qubit_rightmost = qubits[-1].split("]")[0]

            # Add Hadamard gate before the CCZ (now CCX)
            h_before = f"h q[{qubit_rightmost}];"
            transformed_lines.append(h_before)

            # Replace ccz with ccx
            new_ccx_line = line.replace("ccz", "ccx")
            transformed_lines.append(new_ccx_line)

            # Add Hadamard gate after the CCX
            h_after = f"h q[{qubit_rightmost}];"
            transformed_lines.append(h_after)
        else:
            # If no ccz, just append the line
            transformed_lines.append(line)

    # Join the lines back into a single string
    transformed_qasm_code = "\n".join(transformed_lines)
    return transformed_qasm_code


def qasm_file_transpiler(filepath_in, filepath_out):
    circ = load(filepath_in)
    # name = pl.Path(filepath_in).name
    circ_opt = transpile(circ, optimization_level=0, basis_gates=["h", "t", "tdg", "sdg", "s", "x", "y", "z", "cx"])
    dump(circ_opt, filepath_out)
