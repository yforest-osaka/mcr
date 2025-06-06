import os
import pickle

import pyzx as zx
from qulacs import QuantumCircuit
from qulacs.converter import convert_qulacs_circuit_to_QASM


def mkdir_tmp_if_not_exists():
    """Create the 'tmp' directory if it does not exist"""
    if not os.path.exists("tmp"):
        os.makedirs("tmp", exist_ok=True)


def qulacs_to_qasm(filepath: str, qulacs_circ: QuantumCircuit) -> None:
    """Save a Qulacs QuantumCircuit to a QASM file

    Args:
        filepath (str): Path to save the file
        qulacs_circ (QuantumCircuit): Qulacs QuantumCircuit to save
    """
    string = convert_qulacs_circuit_to_QASM(qulacs_circ)
    with open(filepath, mode="w") as f:
        f.write("\n".join(string))


def qasm_to_pyzx(filepath: str) -> zx.Circuit:
    """Convert a QASM file to a PyZX Circuit

    Args:
        filepath (str): Path to the QASM file

    Returns:
        zx.Circuit: PyZX Circuit
    """
    return zx.Circuit.load(filepath)


def qasm_file_to_qc(input_file: str, output_file: str) -> None:
    """Convert a QASM file to a QC file (via PyZX)

    Args:
        input_file (str): Path to the QASM file
        output_file (str): Path to the QC file
    """

    input_circuit = zx.Circuit.from_qasm_file(input_file)
    qc_output = input_circuit.to_qc().replace("Tof", "tof")
    with open(output_file, mode="w") as f:
        f.write(qc_output)


def save_by_pickle(filepath: str, list_data: list) -> None:
    """Save list data in pickle format to the specified file path

    Args:
        filepath (str): File path to save the data
        list_data (list): List data to save
    """
    with open(filepath, mode="wb") as fo:
        pickle.dump(list_data, fo)
