from typing import List, Tuple
from itertools import combinations
import numpy as np
from qulacs import QuantumCircuit
from qiskit import QuantumCircuit as QiskitCircuit
from qulacs.gate import RotZ, S, Sdag, H, X, Y, Z, CNOT, CZ
from qiskit.circuit.library import (
    RZZGate,
    RZGate,
    CXGate,
    CZGate,
    HGate,
    SGate,
    SdgGate,
    XGate,
    YGate,
    ZGate,
)
from pytket import Circuit

# from pytket.passes import CliffordSimp
from pytket.passes import RemoveRedundancies
from pytket.transform import Transform
from qulacs.converter import convert_qulacs_circuit_to_QASM


def apply_pauli_gates(
    circuit_lst, qubit_indices, pauli_ids, right_side=False, use_qulacs=False
):
    """Apply Pauli gates to the specified qubits."""
    pauli_ids = [pauli_ids[i] for i in qubit_indices]
    assert len(pauli_ids) == len(qubit_indices)
    if use_qulacs:
        for idx, pauli_id in zip(qubit_indices, pauli_ids):
            if pauli_id == "X":  # X
                circuit_lst.add_gate(H(idx))
            elif pauli_id == "Y":  # Y
                if right_side:
                    circuit_lst.add_gate(H(idx))
                    circuit_lst.add_gate(S(idx))
                else:
                    circuit_lst.add_gate(Sdag(idx))
                    circuit_lst.add_gate(H(idx))
    else:
        for idx, pauli_id in zip(qubit_indices, pauli_ids):
            if pauli_id == "X":  # X
                circuit_lst.append(("H", [idx]))
            elif pauli_id == "Y":  # Y
                if right_side:
                    circuit_lst.append(("H", [idx]))
                    circuit_lst.append(("S", [idx]))
                else:
                    circuit_lst.append(("Sdg", [idx]))
                    circuit_lst.append(("H", [idx]))


class PauliBit:
    """
    Represents a Pauli operator on multiple qubits with a signed rotation angle.
    Internally stores binary representation of each qubit's X and Z components.
    """

    _pauli_map: dict[str, Tuple[int, int]] = {
        "_": (0, 0),
        "I": (0, 0),
        "X": (1, 0),
        "Z": (0, 1),
        "Y": (1, 1),
    }
    _inv_pauli_map: dict[Tuple[int, int], str] = {
        (0, 0): "I",
        (1, 0): "X",
        (0, 1): "Z",
        (1, 1): "Y",
    }

    def __init__(self, pauli_str: str, angle: float):
        """
        Initialize PauliBit from a Pauli string and an angle.

        Args:
            pauli_str (str): Pauli string like "XZIY" (supports I, X, Y, Z, _).
            angle (float): Rotation angle; must be non-zero.

        Raises:
            ValueError: If pauli_str contains invalid characters or angle is 0.
        """
        pauli_str = pauli_str.upper()
        try:
            self.x_n, self.z_n = zip(*[self._pauli_map[p] for p in pauli_str])
        except KeyError:
            raise ValueError(f"Invalid Pauli string: {pauli_str}")
        self.x_n: List[int] = list(self.x_n)
        self.z_n: List[int] = list(self.z_n)

        # if angle == 0:
        #     raise ValueError("Angle cannot be zero.")
        self.sgn: int = 0 if angle >= 0 else 1
        self.__angle: float = abs(angle)

    def duplicate(self) -> "PauliBit":
        """
        Returns a deep copy of the current PauliBit instance.
        """
        return PauliBit(self.get_pauli_str(), self.get_angle())

    def __repr__(self) -> str:
        # return f"PauliBit(x_n={self.x_n}, z_n={self.z_n}, sgn={self.sgn})"
        return f"PauliBit({self.get_angle()}*{self.get_pauli_str()})"

    def get_pauli_str(self, with_sgn=False) -> str:
        """
        Returns the Pauli string representation from internal x/z vectors.

        Returns:
            str: Pauli string like "IXYZ"
        """
        pauli_str = "".join(
            self._inv_pauli_map[(x, z)] for x, z in zip(self.x_n, self.z_n)
        )
        if with_sgn:
            sgn = np.sign(self.get_angle())
            if sgn == -1:
                return f"-{pauli_str}"
            else:
                return f"+{pauli_str}"
        return pauli_str

    def set_new_angle(self, angle: float) -> None:
        """
        Sets a new angle for the PauliBit.

        Args:
            angle (float): New angle to set.
        """
        self.sgn = 0 if angle >= 0 else 1
        self.__angle = abs(angle)

    def get_angle(self) -> float:
        """
        Returns the signed angle.

        Returns:
            float: Original angle with sign restored.
        """
        return -self.__angle if self.sgn else self.__angle

    def get_data(self) -> Tuple[str, float]:
        """
        Returns the Pauli string and signed angle as a tuple.

        Returns:
            tuple[str, float]: (pauli_str, angle)
        """
        return self.get_pauli_str(), self.get_angle()

    def apply_h(self, qubit_index: int) -> None:
        """
        Applies Hadamard gate to the specified qubit.
        """
        x, z = self.x_n[qubit_index], self.z_n[qubit_index]
        self.x_n[qubit_index], self.z_n[qubit_index] = z, x
        self.sgn ^= x & z  # Y -> -Y

    def apply_x(self, qubit_index: int) -> None:
        """
        Applies X gate to the specified qubit.
        """
        self.sgn ^= self.z_n[qubit_index]

    def apply_y(self, qubit_index: int) -> None:
        """
        Applies Y gate to the specified qubit.
        """
        self.sgn ^= self.x_n[qubit_index] ^ self.z_n[qubit_index]

    def apply_z(self, qubit_index: int) -> None:
        """
        Applies Z gate to the specified qubit.
        """
        self.sgn ^= self.x_n[qubit_index]

    def apply_s(self, qubit_index: int) -> None:
        """
        Applies S (phase) gate to the specified qubit.
        """
        x, z = self.x_n[qubit_index], self.z_n[qubit_index]
        self.z_n[qubit_index] ^= x
        self.sgn ^= x & z

    def apply_sdg(self, qubit_index: int) -> None:
        """
        Applies S† (S dagger) gate to the specified qubit.
        """
        x, z = self.x_n[qubit_index], self.z_n[qubit_index]
        self.z_n[qubit_index] ^= x
        self.sgn ^= x & ~z

    def apply_cz(self, control_qubit_index: int, target_qubit_index: int) -> None:
        """
        Applies CZ gate between control and target qubits.

        Raises:
            AssertionError: If control and target indices are the same.
        """
        assert control_qubit_index != target_qubit_index, (
            "Control and target qubit must be different."
        )
        x_c, z_c = self.x_n[control_qubit_index], self.z_n[control_qubit_index]
        x_t, z_t = self.x_n[target_qubit_index], self.z_n[target_qubit_index]

        self.z_n[target_qubit_index] ^= x_c
        self.z_n[control_qubit_index] ^= x_t

        if x_c & x_t & (z_c ^ z_t):
            self.sgn ^= 1

    def apply_cx(self, control_qubit_index: int, target_qubit_index: int) -> None:
        """
        Applies CX (CNOT) gate between control and target qubits.

        Raises:
            AssertionError: If control and target indices are the same.
        """
        assert control_qubit_index != target_qubit_index, (
            "Control and target qubit must be different."
        )
        x_c, z_c = self.x_n[control_qubit_index], self.z_n[control_qubit_index]
        x_t, z_t = self.x_n[target_qubit_index], self.z_n[target_qubit_index]

        self.x_n[target_qubit_index] ^= x_c
        self.z_n[control_qubit_index] ^= z_t

        if x_c & z_t & ~(x_t ^ z_c):
            self.sgn ^= 1

    def apply_swap(self, qubit_index_1: int, qubit_index_2: int) -> None:
        """
        Swaps the Pauli operators of two qubits.
        """
        self.x_n[qubit_index_1], self.x_n[qubit_index_2] = (
            self.x_n[qubit_index_2],
            self.x_n[qubit_index_1],
        )
        self.z_n[qubit_index_1], self.z_n[qubit_index_2] = (
            self.z_n[qubit_index_2],
            self.z_n[qubit_index_1],
        )

    def clifford_update(self, clifford_gate_sequence) -> None:
        updated_pauli_bit = self.duplicate()
        for gate_info in clifford_gate_sequence:
            gate_name, qubit_indices = gate_info
            if gate_name == "CNOT":
                updated_pauli_bit.apply_cx(qubit_indices[0], qubit_indices[1])
            elif gate_name == "H":
                updated_pauli_bit.apply_h(qubit_indices[0])
            elif gate_name == "S":
                updated_pauli_bit.apply_s(qubit_indices[0])
            elif gate_name == "Sdg":
                updated_pauli_bit.apply_sdg(qubit_indices[0])
            elif gate_name == "X":
                updated_pauli_bit.apply_x(qubit_indices[0])
            elif gate_name == "Y":
                updated_pauli_bit.apply_y(qubit_indices[0])
            elif gate_name == "Z":
                updated_pauli_bit.apply_z(qubit_indices[0])
            elif gate_name == "CZ":
                updated_pauli_bit.apply_cz(qubit_indices[0], qubit_indices[1])
            elif gate_name == "SWAP":
                updated_pauli_bit.apply_swap(qubit_indices[0], qubit_indices[1])
            else:
                raise ValueError(f"Unknown gate: {gate_name}")
        return updated_pauli_bit

    def commutes(self, other: "PauliBit") -> bool:
        """
        Checks if two Pauli operators commute.

        Args:
            other (PauliBit): Another PauliBit instance.

        Returns:
            bool: True if they commute, False otherwise.
        """
        # Two Paulis commute iff the sum over all qubits of (x1*z2 - z1*x2) is even
        comm = sum(
            (x1 * z2 - z1 * x2)
            for x1, z1, x2, z2 in zip(self.x_n, self.z_n, other.x_n, other.z_n)
        )
        return comm % 2 == 0

    def is_clifford(self) -> bool:
        """
        Checks if the Pauli operator is a Clifford operation.

        Returns:
            bool: True if it is a Clifford operation, False otherwise.
        """
        modulus = (self.get_angle() / (np.pi / 4)) % 8
        if np.allclose(modulus, int(modulus)) and int(modulus) % 2 == 0:
            return True
        return False

    def get_non_identity_pauli_indices(self) -> List[int]:
        strs = self.get_pauli_str()
        return [i for i, s in enumerate(strs) if s != "I"]

    def get_clifford_gate_sequence(self) -> List[str]:
        """
        Returns the sequence of Clifford gates that can be applied to this Pauli operator.

        Returns:
            List[str]: List of Clifford gate names.
        """
        assert self.is_clifford(), (
            f"Not a Clifford operation: {self.get_angle()}*{self.get_pauli_str()}"
        )
        pauli_str = self.get_pauli_str()
        modulus = int((self.get_angle() / (np.pi / 4)) % 8)

        non_identity_pauli_indices = self.get_non_identity_pauli_indices()
        position = max(non_identity_pauli_indices)  # The last non-zero index
        circuit = []
        apply_pauli_gates(
            circuit, non_identity_pauli_indices, pauli_str, right_side=False
        )

        if len(non_identity_pauli_indices) >= 2:  # CNOT is required
            # Apply CNOT gates
            for idx in non_identity_pauli_indices:
                if idx != position:
                    circuit.append(("CNOT", [idx, position]))

        # Apply rotation gates(depending on the modulus)
        if modulus == 2:
            circuit.append(("S", [position]))
        elif modulus == 6:
            circuit.append(("Sdg", [position]))
        elif modulus == 4:
            circuit.append(("Z", [position]))

        if len(non_identity_pauli_indices) >= 2:  # CNOT is required
            # Apply CNOT gates in reverse order
            for idx in reversed(non_identity_pauli_indices):
                if idx != position:
                    circuit.append(("CNOT", [idx, position]))
        # Apply Pauli gates in reverse order
        apply_pauli_gates(
            circuit, non_identity_pauli_indices, pauli_str, right_side=True
        )
        return circuit

    def convert_into_qulacs(self) -> List[str]:
        """
        Returns the sequence of Clifford gates that can be applied to this Pauli operator.

        Returns:
            List[str]: List of Clifford gate names.
        """
        pauli_str = self.get_pauli_str()

        # modulus = int((self.get_angle() / (np.pi / 4)) % 8)

        non_identity_pauli_indices = self.get_non_identity_pauli_indices()
        position = max(non_identity_pauli_indices)  # The last non-zero index
        circuit = QuantumCircuit(len(pauli_str))
        apply_pauli_gates(
            circuit,
            non_identity_pauli_indices,
            pauli_str,
            right_side=False,
            use_qulacs=True,
        )

        if len(non_identity_pauli_indices) >= 2:  # CNOT is required
            # Apply CNOT gates
            for idx in non_identity_pauli_indices:
                if idx != position:
                    circuit.add_gate(CNOT(idx, position))

        # Apply rotation gates(depending on the modulus)
        circuit.add_gate(RotZ(position, self.get_angle()))

        if len(non_identity_pauli_indices) >= 2:  # CNOT is required
            # Apply CNOT gates in reverse order
            for idx in reversed(non_identity_pauli_indices):
                if idx != position:
                    circuit.add_gate(CNOT(idx, position))
        # Apply Pauli gates in reverse order
        apply_pauli_gates(
            circuit,
            non_identity_pauli_indices,
            pauli_str,
            right_side=True,
            use_qulacs=True,
        )
        return circuit

    def convert_into_qasm_str(self, with_header=False) -> None:
        qulacs_circuit = self.convert_into_qulacs()
        string = convert_qulacs_circuit_to_QASM(qulacs_circuit)
        if with_header:
            return string
        return string[3:]


def multiply_all(pauli_bit_lst: List["PauliBit"]) -> Tuple[float, str]:
    coef = 1
    pauli_str = ""
    nqubits = len(pauli_bit_lst[0].x_n)
    x_n = [0 for _ in range(nqubits)]
    z_n = [0 for _ in range(nqubits)]
    for pb in pauli_bit_lst:
        if not isinstance(pb, PauliBit):
            raise TypeError(f"Expected PauliBit, got {type(pb)}")
        new_x_n = [(x1 ^ x2) for x1, x2 in zip(x_n, pb.x_n)]
        new_z_n = [(z1 ^ z2) for z1, z2 in zip(z_n, pb.z_n)]
        if pb.sgn == 1:  # negative case
            coef *= -1
        s = 0
        for x_1, z_1, x_2, z_2, x_3, z_3 in zip(
            x_n, z_n, pb.x_n, pb.z_n, new_x_n, new_z_n
        ):
            s += 2 * x_1 * z_2 + z_1 * x_1 + z_2 * x_2 - z_3 * x_3
        s %= 4
        coef *= [1, -1j, -1, 1j][s]
        x_n, z_n = new_x_n, new_z_n

    for x, z in zip(x_n, z_n):
        if x == 0 and z == 0:
            pauli_str += "I"
        elif x == 1 and z == 0:
            pauli_str += "X"
        elif x == 0 and z == 1:
            pauli_str += "Z"
        elif x == 1 and z == 1:
            pauli_str += "Y"
    return coef, pauli_str


def grouping(pauli_bit_data_lst):
    L = []  # Empty list L
    for Rp in pauli_bit_data_lst:
        # print('Rp', Rp)
        j = 0  # Initialize (create a new layer if no anti-commuting group is found)
        for k in reversed(range(len(L))):
            commute_info = [Rp.commutes(Rk) for Rk in L[k]]
            if not all(commute_info):  # If there is even one anti-commuting element
                j = k + 1
                break
            else:
                pass
        if j == 0:
            if len(L) == 0:
                L.append([Rp])
            else:
                L[0].append(Rp)
        else:
            if len(L) == j:
                # Create a new layer
                L.append([Rp])
            else:
                # Add to an existing group
                L[j].append(Rp)
    # for group in L:
    #     commute_check = [ele1.commutes(ele2) for ele1, ele2 in combinations(group, 2)]
    #     if not all(commute_check):
    #         raise ValueError(f"Grouping failed: some elements do not commute.: {group}")
    return L


def synthesize_sequence(pauli_bit_data_lst):
    length = len(pauli_bit_data_lst)
    if length == 1:
        return pauli_bit_data_lst
    pauli_bit_data_lst = sorted(pauli_bit_data_lst, key=lambda x: x.get_pauli_str())
    results = []
    target_str = pauli_bit_data_lst[0].get_pauli_str()
    angle = pauli_bit_data_lst[0].get_angle()
    for idx, elem in enumerate(pauli_bit_data_lst[1:]):
        if target_str == elem.get_pauli_str():
            angle += elem.get_angle()
            if idx == length - 2 and angle != 0:  # last element must be added
                results.append(PauliBit(target_str, angle))
        else:
            if angle != 0:
                results.append(PauliBit(target_str, angle))
            target_str = elem.get_pauli_str()
            angle = elem.get_angle()
            if idx == length - 2:  # last element must be added
                results.append(PauliBit(target_str, angle))
    return results


def separate_clifford_and_rotation(pauli_bit_data_lst):
    clifford_gates = []
    rotation_gates = []
    for elem in pauli_bit_data_lst:
        if elem.is_clifford():
            clifford_gates += elem.get_clifford_gate_sequence()
        else:
            # angleがpi/2を超えるもの、もしくは-np.pi/2を下回るものは分割してCliffordとnon-Cliffordに分けてappendしたい
            angle = elem.get_angle()
            if angle > np.pi / 2:
                assert angle < np.pi
                residue_angle = angle - np.pi / 2
                clifford_gates += PauliBit(
                    elem.get_pauli_str(), np.pi / 2
                ).get_clifford_gate_sequence()
                rotation_gates.append(PauliBit(elem.get_pauli_str(), residue_angle))
                # 分割してCliffordとnon-Cliffordに分ける
            elif angle < -np.pi / 2:
                assert angle > -np.pi
                residue_angle = angle + np.pi / 2
                clifford_gates += PauliBit(
                    elem.get_pauli_str(), -np.pi / 2
                ).get_clifford_gate_sequence()
                rotation_gates.append(PauliBit(elem.get_pauli_str(), residue_angle))
            else:
                rotation_gates.append(elem)
    return clifford_gates, rotation_gates


def get_rotation_relation(pauli_bit_1: PauliBit, pauli_bit_2: PauliBit):
    if pauli_bit_1.get_pauli_str() == pauli_bit_2.get_pauli_str():
        # 回転軸が等しい場合
        sum_angle = pauli_bit_1.get_angles() + pauli_bit_2.get_angles()
        modulus = int((sum_angle / (np.pi / 4)) % 8)
        if np.allclose(
            modulus, int(modulus)
        ):  # Check if modulus is an integer (if true, exactly decomposed into Clifford+T)
            return int(modulus)
        else:
            return "synthesize"
    elif pauli_bit_1.commutes(pauli_bit_2):
        return "commute"
    else:
        return "anti-commute"


def zhang_optimization(pauli_bit_data_lst: List[PauliBit]):
    # ここに入ってくるdataはどれもcommuteするような集合。欲しいのはClifford(clifford_circuit)とnon-Cliffordの列(optimized_rotations)

    rots = synthesize_sequence(pauli_bit_data_lst)
    clifford_gate_sequence, optimized_rotations = separate_clifford_and_rotation(rots)

    return optimized_rotations, clifford_gate_sequence


def loop_optimization(pauli_bit_list, show_log=True):
    flag = True
    length = len(pauli_bit_list)
    initial_length = length
    clifford_data = []
    k = 0
    while flag:
        updated_rots = []
        grouping_data = grouping(pauli_bit_list)
        non_clifford_rots_group, extracted_clifford_group = [], []
        for elem in grouping_data:
            tmp = zhang_optimization(elem)
            non_clifford_rots_group.append(tmp[0])
            extracted_clifford_group.append(tmp[1])
        if (
            sum(non_clifford_rots_group, []) != []
        ):  # Cliffordを外側に移すためにnon-Cliffordをupdate
            for i in range(len(non_clifford_rots_group)):
                clifford_seq = extracted_clifford_group[i]
                if len(clifford_seq) > 0 and len(updated_rots) > 0:
                    updated_rots = [
                        ele.clifford_update(clifford_gate_sequence=clifford_seq)
                        for ele in updated_rots
                    ]
                    updated_rots += non_clifford_rots_group[i]
                    clifford_data += clifford_seq
                else:
                    updated_rots += non_clifford_rots_group[i]
                    clifford_data += clifford_seq
        else:  # non-Cliffordがの個数が0になり、Cliffordだけが残っている場合
            for clifford in extracted_clifford_group:
                clifford_data += clifford
        if len(updated_rots) < length:
            k += 1
            length = len(updated_rots)
            pauli_bit_list = updated_rots
        else:
            if show_log and k > 0:
                print("=" * 40)
                print(f"{k}-iteration optimization applied!")
                print(f"optimization result: {initial_length} -> {len(updated_rots)}")
                print("=" * 40)
            flag = False

    return clifford_data, updated_rots


def set_clifford_to_qulacs(qulacs_circuit, data_lst):
    for elem in data_lst:
        gate_name, qubit_indices = elem
        if gate_name == "CNOT":
            qulacs_circuit.add_gate(CNOT(qubit_indices[0], qubit_indices[1]))
        elif gate_name == "CZ":
            qulacs_circuit.add_gate(CZ(qubit_indices[0], qubit_indices[1]))
        elif gate_name == "H":
            qulacs_circuit.add_gate(H(qubit_indices[0]))
        elif gate_name == "S":
            qulacs_circuit.add_gate(S(qubit_indices[0]))
        elif gate_name == "Sdg":
            qulacs_circuit.add_gate(Sdag(qubit_indices[0]))
        elif gate_name == "X":
            qulacs_circuit.add_gate(X(qubit_indices[0]))
        elif gate_name == "Y":
            qulacs_circuit.add_gate(Y(qubit_indices[0]))
        elif gate_name == "Z":
            qulacs_circuit.add_gate(Z(qubit_indices[0]))
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    return qulacs_circuit


def set_clifford_to_qiskit(data_lst, num_qubits):
    qiskit_circuit = QiskitCircuit(num_qubits)
    for elem in data_lst:
        gate_name, qubit_indices = elem
        if gate_name == "CNOT":
            qiskit_circuit.append(CXGate(), qubit_indices)
        elif gate_name == "CZ":
            qiskit_circuit.append(CZGate(), qubit_indices)
        elif gate_name == "H":
            qiskit_circuit.append(HGate(), qubit_indices)
        elif gate_name == "S":
            qiskit_circuit.append(SGate(), qubit_indices)
        elif gate_name == "Sdg":
            qiskit_circuit.append(SdgGate(), qubit_indices)
        elif gate_name == "X":
            qiskit_circuit.append(XGate(), qubit_indices)
        elif gate_name == "Y":
            qiskit_circuit.append(YGate(), qubit_indices)
        elif gate_name == "Z":
            qiskit_circuit.append(ZGate(), qubit_indices)
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    return qiskit_circuit


def set_clifford_to_tket(tket_circuit, data_lst):
    for elem in data_lst:
        gate_name, qubit_indices = elem
        if gate_name == "CNOT":
            tket_circuit.CX(qubit_indices[0], qubit_indices[1])
        elif gate_name == "CZ":
            tket_circuit.CZ(qubit_indices[0], qubit_indices[1])
        elif gate_name == "H":
            tket_circuit.H(qubit_indices[0])
        elif gate_name == "S":
            tket_circuit.S(qubit_indices[0])
        elif gate_name == "Sdg":
            tket_circuit.Sdg(qubit_indices[0])
        elif gate_name == "X":
            tket_circuit.X(qubit_indices[0])
        elif gate_name == "Y":
            tket_circuit.Y(qubit_indices[0])
        elif gate_name == "Z":
            tket_circuit.Z(qubit_indices[0])
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    return tket_circuit


def clifford_optimization_using_tket(tket_clifford_circuit):
    result_circuit = tket_clifford_circuit.copy()
    RemoveRedundancies().apply(result_circuit)
    # Transform.OptimiseCliffords().apply(result_circuit)
    # Transform.RebaseToCliffordSingles().apply(result_circuit)
    return result_circuit


def convert_rzz_or_rz_set(pauli_bit_lst):
    pauli_ids = pauli_bit_lst.get_pauli_str()
    qiskit_circuit = QiskitCircuit(len(pauli_ids))
    pauli_ids = [pauli_id for pauli_id in pauli_ids if pauli_id != "I"]
    angle = pauli_bit_lst.get_angle()
    qubit_indices = pauli_bit_lst.get_non_identity_pauli_indices()
    assert len(qubit_indices) in {1, 2}, (
        f"Only single or double qubit gates are supported: {len(qubit_indices)} qubits"
    )
    for idx, pauli_id in zip(qubit_indices, pauli_ids):
        if pauli_id == "X":  # X
            qiskit_circuit.append(HGate(), [idx])
        elif pauli_id == "Y":  # Y
            qiskit_circuit.append(SdgGate(), [idx])
            qiskit_circuit.append(HGate(), [idx])
    if len(pauli_ids) == 2:
        qiskit_circuit.append(RZZGate(angle), qubit_indices)
    elif len(pauli_ids) == 1:
        qiskit_circuit.append(RZGate(angle), [qubit_indices[-1]])
    for idx, pauli_id in zip(qubit_indices, pauli_ids):
        if pauli_id == "X":  # X
            qiskit_circuit.append(HGate(), [idx])
        elif pauli_id == "Y":  # Y
            qiskit_circuit.append(HGate(), [idx])
            qiskit_circuit.append(SGate(), [idx])
    return qiskit_circuit
