import os
import re
from uuid import uuid4
import numpy as np

from mqt import qcec
from mqt.qcec.configuration import augment_config_from_kwargs
from mqt.qcec.pyqcec import Configuration
from qulacs import QuantumCircuit

from mcr.filesave import qulacs_to_qasm


def get_qubit_count_from_qasm_file(filepath):
    # Read the file
    with open(filepath, "r") as file:
        content = file.read()

    # Use regular expressions to extract the number next to qreg
    match = re.search(r"qreg q\[(\d+)\];", content)
    if match:
        number = match.group(1)
        return int(number)
    else:
        raise ValueError("Could not determine the number of qubits")


def equivalence_check_via_mqt_qcec(
    circuit_1, circuit_2, exclude_zx_checker=False, show_log=True
):
    # circuit_1, circuit_2 can also be QASM files
    qubit_limit = 25
    remove_flag1, remove_flag2 = False, False
    if isinstance(circuit_1, QuantumCircuit):
        if circuit_1.get_qubit_count() > qubit_limit:
            print("Skipped: too large qubit")
            return True
        filepath1 = f"qcec_tmp1_{uuid4()}.qasm"
        qulacs_to_qasm(filepath1, circuit_1)
        circuit_1 = filepath1
        remove_flag1 = True
    if isinstance(circuit_2, QuantumCircuit):
        if circuit_2.get_qubit_count() > qubit_limit:
            print("Skipped: too large qubit")
            return True
        filepath2 = f"qcec_tmp2_{uuid4()}.qasm"
        qulacs_to_qasm(filepath2, circuit_2)
        circuit_2 = filepath2
        remove_flag2 = True

    if (
        get_qubit_count_from_qasm_file(circuit_1) > qubit_limit
        and get_qubit_count_from_qasm_file(circuit_2) > qubit_limit
    ):
        print("Skipped: too large qubit")
        return True
    if exclude_zx_checker:
        configuration = Configuration()
        augment_config_from_kwargs(
            configuration,
            {"run_simulation_checker": True, "run_zx_checker": False, "timeout": 5},
        )
        result = qcec.verify(
            circuit_1, circuit_2, configuration=configuration
        ).equivalence
    else:
        result = qcec.verify(circuit_1, circuit_2).equivalence
    if remove_flag1:
        os.remove(circuit_1)
    if remove_flag2:
        os.remove(circuit_2)
    if show_log:
        print(result.name)
    return result.name in {"equivalent", "equivalent_up_to_global_phase"}


def pauli_bit_equivalence_check(pauli_bit_lst_1, pauli_bit_lst_2):
    from mcr.gate_apply import PauliBit

    if pauli_bit_lst_1:
        nqubits = len(pauli_bit_lst_1[0].get_pauli_str())
    elif pauli_bit_lst_2:
        nqubits = len(pauli_bit_lst_2[0].get_pauli_str())
    else:
        # raise ValueError("Both lists of PauliBits are empty, cannot determine nqubits")
        return True  # Both lists are empty, considered equivalent
    if len(pauli_bit_lst_1) == 0:
        pauli_bit_lst_1 = [
            PauliBit("Z" * nqubits, np.pi / 4),
            PauliBit("Z" * nqubits, -np.pi / 4),
        ]
    if len(pauli_bit_lst_2) == 0:
        pauli_bit_lst_2 = [
            PauliBit("Z" * nqubits, np.pi / 4),
            PauliBit("Z" * nqubits, -np.pi / 4),
        ]
    circuit_1, circuit_2 = QuantumCircuit(nqubits), QuantumCircuit(nqubits)
    for elem in pauli_bit_lst_1:
        circuit_1.merge_circuit(elem.convert_into_qulacs())
    for elem in pauli_bit_lst_2:
        circuit_2.merge_circuit(elem.convert_into_qulacs())
    return equivalence_check_via_mqt_qcec(
        circuit_1, circuit_2, exclude_zx_checker=True, show_log=False
    )


def equiv(seq_1: list, seq_2: list):
    from mcr.gate_apply import set_clifford_to_qulacs

    clifford_lst_1, non_clifford_pauli_bits_1 = seq_1
    clifford_lst_2, non_clifford_pauli_bits_2 = seq_2
    if len(non_clifford_pauli_bits_1) > 0:
        nqubits = len(non_clifford_pauli_bits_1[0].get_pauli_str())
    else:
        nqubits = len(non_clifford_pauli_bits_2[0].get_pauli_str())
    if nqubits > 10:
        return True
    circuit_1, circuit_2 = QuantumCircuit(nqubits), QuantumCircuit(nqubits)
    if len(clifford_lst_1) > 0:
        circuit_1 = set_clifford_to_qulacs(circuit_1, clifford_lst_1)
    if len(non_clifford_pauli_bits_1) > 0:
        for elem in non_clifford_pauli_bits_1:
            circuit_1.merge_circuit(elem.convert_into_qulacs())
    if len(clifford_lst_2) > 0:
        circuit_2 = set_clifford_to_qulacs(circuit_2, clifford_lst_2)
    if len(non_clifford_pauli_bits_2) > 0:
        for elem in non_clifford_pauli_bits_2:
            circuit_2.merge_circuit(elem.convert_into_qulacs())
    return equivalence_check_via_mqt_qcec(
        circuit_1, circuit_2, exclude_zx_checker=True, show_log=False
    )
