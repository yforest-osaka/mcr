# 基本ライブラリ
import collections
import copy
import math
from typing import List

# プロット
import matplotlib.pyplot as plt
import numpy as np
import stim

# Qulacs関連
from qulacs import QuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
from qulacs.converter import convert_qulacs_circuit_to_QASM
from qulacs.gate import CNOT, H, Identity, PauliRotation, S, Sdag, T, Tdag, X, Y, Z
from qulacsvis import circuit_drawer as draw

from mcr.equiv_check import equivalence_check_via_mqt_qcec


class PauliRotationSequence:
    def __init__(self, n_qubit: int) -> None:
        """Initializes a quantum program with a specified number of qubits.

        Args:
            n_qubit (int): The number of qubits in the quantum program.
        """
        self.__gate_list = []
        self.gate_id_list = []
        self.del_nums = set()
        self.index_distribution = {f"{i}": 0 for i in range(n_qubit)}
        self.__n_qubit: int = n_qubit

    def add_gate(self, gate_id: tuple[int], gate: stim.PauliString) -> None:
        """
        Adds a gate to the quantum program.

        Args:
            gate_id (tuple[int]): A tuple representing the gate identifier.
            gate (stim.PauliString): The gate to be added, which must be an instance of a stim.PauliString.

        Raises:
            ValueError: If the gate is not an instance of stim.PauliString.

        Updates:
            self.__gate_list: Appends the gate_id and gate as a tuple.
            self.index_distribution: Increments the count for each index in the gate's target index list.
        """
        if isinstance(gate, str):
            gate = stim.PauliString(gate)

        assert isinstance(
            gate, stim.PauliString
        ), "The gate must be an instance of a stim.PauliString."
        pauli_string_length = len(gate)
        diff = self.__n_qubit - pauli_string_length
        if diff > 0:
            tmp = str(gate)
            tmp += "_" * diff
            gate = stim.PauliString(tmp)
        self.__gate_list.append((gate_id, gate))
        self.gate_id_list.append(gate_id)
        indices = gate.pauli_indices()
        for index in indices:
            self.index_distribution[f"{index}"] += 1  # Update index_distribution

    def __getitem__(self, position: int) -> tuple:
        """Defines behavior for circuit[position].

        Args:
            position (int): The position of the gate to retrieve.

        Returns:
            tuple: (gate_id, gate)
        """
        return self.__gate_list[
            position
        ]  # Retrieve the element at the specified position.

    def __len__(self) -> int:
        """Defines behavior for len() (note: includes gates marked for deletion).

        Returns:
            int: The number of gates (including those marked for deletion).
        """
        return len(self.__gate_list)

    def list(self) -> list:
        """Defines behavior for list() (note: includes gates marked for deletion).

        Returns:
            list: The list of gates (including those marked for deletion).
        """
        return self.__gate_list

    def get_qubit_count(self) -> int:
        """Retrieves the number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self.__n_qubit

    def get_index_distribution(self, show_graph: bool = False) -> collections.Counter:
        """Retrieves the distribution of qubits involved in gates.

        Args:
            show_graph (bool, optional): Whether to display a graph. Defaults to False.

        Returns:
            collections.Counter: The distribution of qubits involved in gates (Counter).
        """
        results = []
        for g in self.sort_gate_sequence():
            gate = g[1]
            for index in gate.pauli_indices():
                results.append(index)
        data = collections.Counter(results)
        if show_graph:
            plt.bar(list(data.keys()), list(data.values()), align="center")
            plt.xlabel("index")
            plt.ylabel("gate_count")
        return collections.Counter(
            self.index_distribution
        )  # Works even if there are qubits without gates.

    def get_all(
        self,
        target_id: tuple | bool = False,
        only_id: bool = False,
    ) -> List:
        """Retrieves all gates in the sequence.

        Args:
            target_id (tuple | NoneType, optional): Retrieve only gates belonging to the specified ID list. Defaults to None.
            only_id (bool, optional): Retrieve only gates matching the specified ID. Defaults to False.

        Returns:
            list: The list of gates.
        """
        datalist = []
        positions = []
        if not target_id:
            for i, data in enumerate(self.__gate_list):
                if i not in self.del_nums:
                    if only_id:
                        datalist.append(data[0])
                        positions.append(i)
                    else:
                        datalist.append(data)
                        positions.append(i)
        else:
            target_id = list(target_id)
            l = len(target_id)
            for i, data in enumerate(self.__gate_list):
                if i not in self.del_nums:
                    if target_id == list(data[0])[:l]:
                        if only_id:
                            datalist.append(data[0])
                            positions.append(i)
                        else:
                            datalist.append(data)
                            positions.append(i)
        return sorted(datalist, key=lambda x: x[0])

    def get_all_ids(self) -> List[tuple[int]]:
        """Retrieves the IDs of all gates.

        Returns:
            list[tuple[int]]: The list of gate IDs.
        """
        assert [
            data[0] for i, data in enumerate(self.__gate_list) if i not in self.del_nums
        ] == [
            ele for i, ele in enumerate(self.gate_id_list) if i not in self.del_nums
        ], "The gate_id_list is not consistent with the gate_list!"
        return [
            data[0] for i, data in enumerate(self.__gate_list) if i not in self.del_nums
        ]

    def delete(self, position: int) -> None:
        """Deletes a gate at the specified position.

        Args:
            position (int): The position of the gate to delete.

        Raises:
            ValueError: If the gate has already been deleted.
        """
        if position in self.del_nums:
            raise ValueError(
                f"The position {position} is already included in del_nums!:{self.del_nums}"
            )
        self.del_nums.add(position)
        # Update index_distribution to reflect the deletion.
        indices = self.__gate_list[position][1].pauli_indices()
        for index in indices:
            self.index_distribution[f"{index}"] -= 1  # Update index_distribution.

    def delete_from_gate_id(self, gate_id: tuple) -> None:
        flag = True
        for idx, ele in enumerate(self.__gate_list):
            if flag:
                if ele[0] == gate_id:
                    self.delete(idx)
                    flag = False
        if flag:
            raise ValueError(f"The gate_id {gate_id} is not included in the gate_list!")

    def sort_gate_sequence(self, only_gates=False) -> List:
        """Sorts the gate sequence in the original order.

        Returns:
            List: The sorted list of gates.
        """
        if only_gates:
            return [ele[1] for ele in sorted(self.get_all())]
        else:
            return sorted(self.get_all())

    def set_circuit(self) -> QuantumCircuit:
        """Sets the circuit in Qulacs.

        Returns:
            QuantumCircuit: The Qulacs circuit.
        """
        nqubits = self.get_qubit_count()
        circuit = QuantumCircuit(nqubits)

        for gate in self.get_all():
            stim_gate_info = gate[1]
            pauli_ids = get_pauli_id_from_stim(stim_gate_info)
            angle = np.pi / 4 if stim_gate_info.sign == 1 else -np.pi / 4

            qubit_indices = [i for i, p in enumerate(pauli_ids) if p != 0]
            non_identity_pauli_ids = [pauli_ids[i] for i in qubit_indices]
            position = find_last_non_zero_index(pauli_ids)

            apply_pauli_gates(
                circuit, qubit_indices, non_identity_pauli_ids, right_side=False
            )

            if len(qubit_indices) >= 2:
                for idx in qubit_indices:
                    if idx != position:
                        circuit.add_gate(CNOT(idx, position))

            apply_rotation_gate(circuit, angle, position)

            if len(qubit_indices) >= 2:
                for idx in reversed(qubit_indices):
                    if idx != position:
                        circuit.add_gate(CNOT(idx, position))

            apply_pauli_gates(
                circuit, qubit_indices, non_identity_pauli_ids, right_side=True
            )

        return circuit

    def draw(self):
        """Draws the quantum circuit."""
        qulacs_circuit = self.set_circuit()
        return draw(qulacs_circuit, "mpl")

    def get_inversed_gates(self) -> List:
        """Retrieves the inverse circuit.

        Returns:
            List: The inverse circuit.
        """
        qulacs_circuit = self.set_circuit()
        inverse = qulacs_circuit.get_inverse()
        gates = []
        for i in range(inverse.get_gate_count()):
            gates.append(inverse.get_gate(i))
        return gates

    def get_gate(self, position: int):
        """Retrieves the gate at the specified position.

        Args:
            position (int): The position of the gate to retrieve.

        Returns:
            QulacsGate: The Qulacs gate.
        """
        data = self.sort_gate_sequence()
        return data[position]

    def get_gate_from_gate_id(self, gate_id: tuple):
        data = self.sort_gate_sequence()
        for ele in data:
            if ele[0] == gate_id:
                return ele[1]
        return None

    def get_gate_count(self) -> int:
        """Retrieves the number of gates.

        Returns:
            int: The number of gates.
        """
        t = len(self.sort_gate_sequence())
        return t

    def is_equivalent(self, another_circuit) -> float:
        """Checks if two quantum circuits are equivalent."""

        qulacs_circuit1 = self.set_circuit()
        qulacs_circuit2 = another_circuit.set_circuit()
        exclude_zx_checker = True
        if equivalence_check_via_mqt_qcec(
            qulacs_circuit1, qulacs_circuit2, exclude_zx_checker, show_log=False
        ):
            return True
        else:
            raise ValueError(f"The two circuits are not equivalent!")

    def duplicate(self) -> "PauliRotationSequence":
        """Duplicates the quantum circuit.

        Returns:
            PauliRotationSequence: The duplicated quantum circuit.
        """
        tmp_circ = PauliRotationSequence(self.__n_qubit)
        tmp_circ.__gate_list = copy.deepcopy(self.__gate_list)  # Copy gate_list.
        tmp_circ.index_distribution = copy.deepcopy(
            self.index_distribution
        )  # Copy index_distribution.
        tmp_circ.del_nums = self.del_nums.copy()  # Copy del_nums.
        return tmp_circ

    def merge(self, block_size: int) -> List:
        """Merges the quantum circuit into blocks of the specified size.

        Args:
            block_size (int): The block size.

        Returns:
            List: The merged quantum circuit.
        """
        set_circ = self.set_circuit()
        tmp = set_circ.copy()
        QCO().optimize(tmp, block_size)
        gates = []
        for i in range(tmp.get_gate_count()):
            gates.append(((i,), tmp.get_gate(i)))
        self.__gate_list = gates
        return self.__gate_list

    def save_qasm(self, filename: str) -> None:
        """Saves the quantum circuit in QASM format.

        Args:
            filename (str): The filename to save to.
        """
        qulacs_circ = self.set_circuit()
        string = convert_qulacs_circuit_to_QASM(qulacs_circ)
        with open(filename, mode="w") as f:
            f.write("\n".join(string))


def complement_identity(circuit: QuantumCircuit) -> QuantumCircuit:
    """Adds Identity gates to qubits that have no gates applied.

    Args:
        circuit (QuantumCircuit): A Qulacs quantum circuit.

    Returns:
        QuantumCircuit: The circuit with Identity gates added.
    """
    indices = []
    for i in range(circuit.get_gate_count()):
        indices += circuit.get_gate(i).get_target_index_list()
        indices += circuit.get_gate(i).get_control_index_list()
    for num in range(circuit.get_qubit_count()):
        if num not in indices:
            circuit.add_gate(Identity(num))
    return circuit


def pauli_bit_to_pauli_id(pauli_bit, with_coef=True):
    """Converts a Pauli bit representation to a Pauli ID.

    Args:
        pauli_bit: The Pauli bit representation.
        with_coef (bool): Whether to include the coefficient. Defaults to True.

    Returns:
        If with_coef is True, returns a tuple of the coefficient and Pauli ID.
        Otherwise, returns only the Pauli ID.
    """
    sgn_dict = {0: 1, 1: 1j, 2: -1, 3: -1j}
    pauli_bit_to_id_dict = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    if with_coef:
        return sgn_dict[pauli_bit[0]], tuple(
            pauli_bit_to_id_dict[i] for i in pauli_bit[1:]
        )
    else:
        return tuple(pauli_bit_to_id_dict[i] for i in pauli_bit[1:])


def pauli_string_to_pauli_bit(pauli_string):
    """Converts a Pauli string to a Pauli bit representation.

    Args:
        pauli_string: The Pauli string to convert.

    Returns:
        A tuple representing the Pauli bit.

    Raises:
        ValueError: If the Pauli string is invalid.
    """
    pauli_str_to_bit_dict = {
        "I": (0, 0),
        "X": (0, 1),
        "Y": (1, 1),
        "Z": (1, 0),
        "_": (0, 0),
    }
    if isinstance(
        pauli_string, tuple
    ):  # If the tuple includes a coefficient (e.g., (-1, 'XX'))
        sgn_dict = {1: 0, 1j: 1, -1: 2, -1j: 3}
        sgn_bit = sgn_dict[pauli_string[0]]
        pauli_string = pauli_string[1]
    else:
        sgn_bit = 0  # Coefficient is 1
    if pauli_string[:2] == "+i":  # If the string starts with +i (for stim usage)
        sgn_bit = 1
        pauli_string = pauli_string[2:]
    elif pauli_string[:2] == "-i":  # If the string starts with -i (for stim usage)
        sgn_bit = 3
        pauli_string = pauli_string[2:]
    elif pauli_string[0] == "-":  # If the string starts with - (for stim usage)
        sgn_bit = 2
        pauli_string = pauli_string[1:]
    elif pauli_string[0] == "+":
        sgn_bit = 0
        pauli_string = pauli_string[1:]
    results = []
    results.append(sgn_bit)
    for pauli in pauli_string:
        try:
            results.append(pauli_str_to_bit_dict[pauli.upper()])
        except KeyError:
            raise ValueError(f"Invalid Pauli String: {pauli}")
    return tuple(results)


def find_last_non_zero_index(tup):
    """Finds the last non-zero index in a tuple.

    Args:
        tup: The input tuple.

    Returns:
        The index of the last non-zero element, or None if all elements are zero.
    """
    for i, val in enumerate(reversed(tup)):
        if val != 0:
            return len(tup) - 1 - i
    return None  # All elements are zero


def apply_pauli_gates(circuit, qubit_indices, pauli_ids, right_side=False):
    """Apply Pauli gates to the specified qubits"""
    for idx, pauli_id in zip(qubit_indices, pauli_ids):
        if pauli_id == 1:  # X
            circuit.add_gate(H(idx))
        elif pauli_id == 2:  # Y
            if right_side:
                circuit.add_gate(H(idx))
                circuit.add_gate(S(idx))
            else:
                circuit.add_gate(Sdag(idx))
                circuit.add_gate(H(idx))


def apply_rotation_gate(circuit, angle, position):
    """Apply T, S, Z, etc., based on the rotation angle"""
    modulus = int((angle / (np.pi / 4)) % 8)
    if modulus == 0:  # I
        pass
    elif modulus == 1:  # T
        circuit.add_gate(T(position))
    elif modulus == 2:  # S
        circuit.add_gate(S(position))
    elif modulus == 3:  # S and T
        circuit.add_gate(S(position))
        circuit.add_gate(T(position))
    elif modulus == 4:  # Z
        circuit.add_gate(Z(position))
    elif modulus == 5:  # Sdag and Tdag
        circuit.add_gate(Sdag(position))
        circuit.add_gate(Tdag(position))
    elif modulus == 6:  # Sdag
        circuit.add_gate(Sdag(position))
    elif modulus == 7:  # Tdag
        circuit.add_gate(Tdag(position))
    else:
        raise ValueError(f"Invalid angle: {angle}")


def get_pauli_id_from_stim(pauli):
    """
    Convert a stim pauli string to a Pauli ID
    """
    pauli_strs = str(pauli)[1:]
    values = []
    for pauli in pauli_strs:
        if pauli == "_":
            values.append(0)
        elif pauli == "X":
            values.append(1)
        elif pauli == "Y":
            values.append(2)
        elif pauli == "Z":
            values.append(3)
        else:
            raise ValueError(f"Unknown Pauli: {pauli}")
    return values
