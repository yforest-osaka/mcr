import re
from pathlib import Path
from time import time

import stim
from joblib import Parallel, delayed
from tqdm import tqdm

from mcr.filesave import save_by_pickle


def process_line(line, pattern):
    """QASMファイルの1行を解析してStimの操作に変換する"""
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

    for ele in circ_qasm:
        if "qreg" in ele:
            qubit_count = int(re.findall(r"\d+", ele)[0])
            break

    exclude = ["OPENQASM", "include", "qreg", "creg", "measure"]
    circ_qasm = [
        line for line in circ_qasm if not any([word in line for word in exclude])
    ]

    # Define a regular expression pattern to match gate names and qubit numbers
    pattern = re.compile(r"(\w+)\s+q\[(\d+)\](?:,q\[(\d+)\])?;")

    stim_circuits_info = []
    circuit = stim.Circuit()

    results = Parallel(n_jobs=-1)(
        delayed(process_line)(line, pattern)
        for line in tqdm(circ_qasm, desc="Extracting gates", leave=False)
    )

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


def stim_litinski_compile(qasm_filepath):
    """
    Compiles a QASM file using the Litinski method and returns the resulting data.

    Args:
        qasm_filepath (str): The file path to the QASM file to be compiled.

    Returns:
        list: A list of strings representing the compiled data. Each string corresponds to a mapped Pauli operator.
    """
    stim_circuits_info, nqubits = qasm_to_stim(qasm_filepath)
    data = []
    used_circuit = stim.Circuit()
    for elem in tqdm(
        reversed(stim_circuits_info), desc="Litinski Compiling", leave=False
    ):
        if isinstance(elem, stim.Circuit):
            target_circuit = elem.copy()
            target_circuit.append_from_stim_program_text(str(used_circuit))
            used_circuit = target_circuit.copy()
        else:
            t_qubit_index, sgn = elem
            try:
                mapped_pauli = str(used_circuit.to_tableau().z_output(t_qubit_index))
            except:
                mapped_pauli = "+" + "_" * nqubits
            if sgn == -1:
                if mapped_pauli[0] == "+":
                    mapped_pauli = "-" + mapped_pauli[1:]
                elif mapped_pauli[0] == "-":
                    mapped_pauli = "+" + mapped_pauli[1:]
            data.append(mapped_pauli)
    data = list(reversed(data))
    return data


def main():
    filepath = "../data/transpiled_qasm/barenco_tof_3.qasm"

    st = time()
    data = stim_litinski_compile(filepath)
    print("Time:", time() - st)
    print(filepath)
    output_filepath = f"../data/litinski_rotations/stim_{Path(filepath).stem}.pickle"
    save_by_pickle(output_filepath, data)


if __name__ == "__main__":
    main()
