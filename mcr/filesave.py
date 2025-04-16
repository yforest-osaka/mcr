import os

import pyzx as zx
from qulacs import QuantumCircuit
from qulacs.converter import convert_qulacs_circuit_to_QASM


def mkdir_tmp_if_not_exists():
    if not os.path.exists("tmp"):
        # Create the 'tmp' directory if it does not exist
        os.makedirs("tmp")


def qulacs_to_qasm(filepath: str, qulacs_circ: QuantumCircuit) -> None:
    """QulacsのQuantumCircuitをQASMファイルに保存

    Args:
        filepath (str): 保存先のファイルパス
        qulacs_circ (QuantumCircuit): 保存するQulacsのQuantumCircuit
    """
    string = convert_qulacs_circuit_to_QASM(qulacs_circ)
    with open(filepath, mode="w") as f:
        f.write("\n".join(string))


def qasm_to_pyzx(filepath: str) -> zx.Circuit:
    """QASMファイルをPyZXのCircuitに変換

    Args:
        filepath (str): QASMファイルのパス

    Returns:
        zx.Circuit: PyZXのCircuit
    """
    return zx.Circuit.load(filepath)


def qasm_file_to_qc(input_file: str, output_file: str) -> None:
    """QASMファイルをQCファイルに変換(PyZX経由)

    Args:
        input_file (str): QASMファイルのパス
        output_file (str): QCファイルのパス
    """

    input_circuit = zx.Circuit.from_qasm_file(input_file)
    qc_output = input_circuit.to_qc().replace("Tof", "tof")
    with open(output_file, mode="w") as f:
        f.write(qc_output)
