import os
import pickle
import re
import uuid
from glob import glob
from uuid import uuid4

import numpy as np
import pyzx as zx
import stim
from joblib import Parallel, delayed
from qulacs import QuantumCircuit
from qulacs.converter import (
    convert_QASM_to_qulacs_circuit,
    convert_qulacs_circuit_to_QASM,
)
from tqdm import tqdm


def string_to_qasm(filepath: str, string: str, join_option: bool = True) -> None:
    """文字列をQASMファイルに保存

    Args:
        filepath (str): 保存先のファイルパス
        string (str): 保存する文字列
        join_option (bool, optional): '\n'をjoinで繋げるかどうか. Defaults to True.
    """
    if join_option:
        with open(filepath, mode="w") as f:
            f.write("\n".join(string))
    else:
        with open(filepath, mode="w") as f:
            f.writelines(string)


def qasm_to_qulacs(filepath: str) -> QuantumCircuit:
    """QASMファイルをQulacsのQuantumCircuitに変換

    Args:
        filepath (str): QASMファイルのパス

    Returns:
        QuantumCircuit: QulacsのQuantumCircuit
    """
    with open(filepath, mode="r") as f:
        circ_qasm = f.read()
    circ_qulacs = convert_QASM_to_qulacs_circuit(circ_qasm.splitlines())
    return circ_qulacs


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


def save_by_pickle(filepath: str, list_data: list) -> None:
    """ファイルパスにリストデータをpickle形式で保存

    Args:
        filepath (str): 保存先のファイルパス
        list_data (list): 保存するリストデータ
    """
    with open(filepath, mode="wb") as fo:
        pickle.dump(list_data, fo)


def read_pickle(filepath: str) -> list:
    """pickle形式のファイルを読み込む

    Args:
        filepath (str): pickle形式のファイルパス

    Returns:
        list: 読み込んだデータのリスト
    """
    with open(filepath, mode="br") as f:
        data = pickle.load(f)
    return data


def save_npy(filepath: str, numpy_data: np.ndarray) -> None:
    """numpyデータを.npy形式で保存

    Args:
        filepath (str): 保存先のファイルパス
        numpy_data (np.ndarray): 保存するnumpyデータ
    """
    np.save(filepath, numpy_data)


def load_npy(filepath: str) -> np.ndarray:
    """npy形式のファイルを読み込む

    Args:
        filepath (str): npy形式のファイルパス

    Returns:
        np.ndarray: 読み込んだデータのnumpyデータ
    """
    return np.load(filepath)


def pyzx_to_qasm(filepath: str, string: str) -> None:
    """pyzxのCircuitをQASMファイルに保存

    Args:
        filepath (str): 保存先のファイルパス
        string (str): 保存するpyzxのCircuit(文字列)
    """
    with open(filepath, mode="w") as f:
        f.write(string)


def delete_qasm_files(directory: str) -> None:
    """qasmファイルを削除

    Args:
        directory (str): 削除するディレクトリ
    """
    # Find all .qasm files in the specified directory
    qasm_files = glob(os.path.join(directory, "*.qasm"))

    # Loop through the list of .qasm files and delete each one
    for file in qasm_files:
        try:
            os.remove(file)
            # print(f'Deleted: {file}')
        except Exception as e:
            print(f"Error deleting {file}: {e}")


def qulacs_to_pyzx(qulacs_circ: QuantumCircuit) -> zx.Circuit:
    """QulacsのQuantumCircuitをPyZXのCircuitに変換

    Args:
        qulacs_circ (QuantumCircuit): QulacsのQuantumCircuit

    Returns:
        zx.Circuit: PyZXのCircuit
    """
    if not os.path.exists("tmp"):
        # Create the 'tmp' directory if it does not exist
        os.makedirs("tmp")
    id = uuid.uuid4()
    qulacs_to_qasm(f"tmp/{id}.qasm", qulacs_circ)
    pyzx_circ = qasm_to_pyzx(f"tmp/{id}.qasm")
    # delete_qasm_files('./tmp')
    os.remove(f"tmp/{id}.qasm")
    return pyzx_circ


def stim_to_qasm(filename, circuit):
    string_to_qasm(filename, circuit.to_qasm(open_qasm_version=2), join_option=False)


def stim_circuit_to_qulacs(circuit):
    stim_to_qasm("tmp.qasm", circuit)
    c = qasm_to_qulacs("tmp.qasm")
    os.remove("tmp.qasm")
    return c


def filter(elem: str):
    if len(elem) == 1 or ".v" in elem or ".i" in elem:
        return False
    elif "begin" in elem.lower() or "end" in elem.lower():
        return False
    return True


def qc_file_to_qasm(input_file: str, output_file: str) -> None:
    """QCファイルをQASMファイルに変換(PyZX経由)

    Args:
        input_file (str): QCファイルのパス
        output_file (str): QASMファイルのパス
    """
    from mcr.converter import qasm_file_transpiler, transform_ccz_to_ccx

    input_circuit = zx.Circuit.from_qc_file(input_file)
    filename = f"{uuid4()}.qasm"
    pyzx_to_qasm(filename, transform_ccz_to_ccx(input_circuit.to_qasm()))  # CCZはCCXに変換しておく
    qasm_file_transpiler(filename, output_file)
    os.remove(filename)


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


def process_line(line):
    """QASMファイルの1行を解析してStimの操作に変換する"""
    # Compile regular expression pattern only once to save overhead.
    pattern = re.compile(r"(\w+)\s+q\[(\d+)\](?:,q\[(\d+)\])?;")
    match = pattern.match(line)
    if match:
        gate_name = match.group(1)
        qubits = [int(match.group(2))]
        if match.group(3):
            qubits.append(int(match.group(3)))
        return gate_name, qubits
    return None, None


def qasm_to_stim(circuit_file):
    with open(circuit_file, mode="r") as f:
        string = f.read().splitlines()
    circ_qasm = [line for line in string if line != ""]

    # Extract qubit count
    for ele in circ_qasm:
        if "qreg" in ele:
            qubit_count = int(re.findall(r"\d+", ele)[0])
            break

    # Exclude irrelevant lines from QASM
    exclude = ["OPENQASM", "include", "qreg", "creg", "measure"]
    circ_qasm = [line for line in circ_qasm if not any([word in line for word in exclude])]

    stim_circuits_info = []
    circuit = stim.Circuit()

    # Use parallel processing to handle parsing
    results = Parallel(n_jobs=-1)(
        delayed(process_line)(line) for line in tqdm(circ_qasm, desc="Extracting gates", leave=False)
    )

    # Append results to Stim Circuit
    for gate_name, qubits in results:
        if gate_name:
            if gate_name not in {"t", "tdg"}:  # Clifford gates
                try:
                    circuit.append(f"{gate_name.upper()}", qubits)
                except:
                    raise ValueError(f"Gate {gate_name} not supported by stim")
            else:  # Non-Clifford gates
                stim_circuits_info.append(circuit)
                circuit = stim.Circuit()
                if gate_name == "t":
                    stim_circuits_info.append([qubits[0], 1])
                else:
                    stim_circuits_info.append([qubits[0], -1])
    if len(circuit) > 0:
        stim_circuits_info.append(circuit)
    return stim_circuits_info, qubit_count
