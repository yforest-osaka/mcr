# N qubit Pauli Rotation Class
import math
from collections import Counter
from itertools import product

import numpy as np
import pyzx as zx
import sympy as sp
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, H, PauliRotation, S, Sdag, T, Tdag, X, Y, Z
from qulacs_core import ClsOneQubitGate
from qulacsvis import circuit_drawer

from mcr.clifford import complement_identity
from mcr.pauli_bit_ops import pauli_bit_to_pauli_id, pauli_string_to_pauli_bit


def find_non_zero_index(tup):
    # タプルを反転して、最初の0でない値を探す
    for i, val in enumerate(reversed(tup)):
        if val != 0:
            # 反転しているため、元のindexに変換
            return len(tup) - 1 - i
    return None  # すべて0の場合


def apply_pauli_gates(circuit, qubit_indices, pauli_ids, right_side=False):
    """指定された量子ビットに対してPauliゲートを適用"""
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
    """回転角に応じてT, S, Zなどを適用"""
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

    ### 旧版 ###
    # if np.allclose(np.abs(angle), np.pi / 4):  # T or Tdag
    #     print(angle)
    #     print(modulus)
    #     assert modulus in [1, 7]
    #     circuit.add_gate(add_T_orTdag(t_id, position))
    # elif np.allclose(np.abs(angle), np.pi / 2):  # S or Sdag
    #     assert modulus in [2, 6]
    #     circuit.add_gate(add_S_orSdag(t_id, position))
    # elif np.allclose(np.abs(angle), np.pi):  # Z
    #     assert modulus in [4]
    #     circuit.add_gate(Z(position))
    # else:
    #     raise ValueError(f"Invalid angle: {angle}")


class RotOps:

    def __init__(
        self, gate_sequence: list[str] | list[list[int]] | list[tuple[int]]
    ) -> None:
        if len(gate_sequence) == 0:
            self.__pauli_bits = []
            self.__angles = []
        else:
            input_element = gate_sequence[0]
            if isinstance(input_element, str):  # ["II", "XX", "II", "YY"]
                self.__pauli_bits = [
                    pauli_string_to_pauli_bit(ele) for ele in gate_sequence
                ]
            else:  # [[0,0], [1,1], [0,0], [2,2]]
                self.__pauli_bits = [
                    pauli_id_to_pauli_bit(ele) for ele in gate_sequence
                ]
            self.__angles = [0 for _ in range(len(gate_sequence))]

    def get_pauli_bits(self):
        return self.__pauli_bits

    def get_pauli_ids(self, include_angle_sgn=False):
        if include_angle_sgn:
            angle_sgns = self.get_sgn_angles()
            data = []
            for pauli_bit, angle_sgn in zip(self.__pauli_bits, angle_sgns):
                if angle_sgn > 0:
                    data.append((1, pauli_bit_to_pauli_id(pauli_bit, with_coef=False)))
                else:
                    data.append((-1, pauli_bit_to_pauli_id(pauli_bit, with_coef=False)))
            return data
        else:
            return [
                pauli_bit_to_pauli_id(ele, with_coef=False) for ele in self.__pauli_bits
            ]

    def get_angles(self):
        return self.__angles

    def get_sgn_angles(self):
        return np.sign(self.__angles)

    def get_pauli_strings(self):
        return [
            pauli_bit_to_pauli_string(ele, with_coef=False) for ele in self.__pauli_bits
        ]

    def get_distribution_pauli(self, minimum_count: int = 1):
        data = Counter(self.get_pauli_strings())
        filtered_data = Counter(
            {key: value for key, value in data.items() if value >= minimum_count}
        )
        return filtered_data

    def get_gate_count(self):
        return len(self.__pauli_bits)

    def get_non_clifford_gate_count(self):
        count = 0
        for angle in self.get_angles():
            multiple = angle / (np.pi / 2)
            if not math.isclose(multiple, round(multiple)):
                count += 1
        return count

    def extract_clifford_part(self):
        new_pauli_bits = []
        new_angles = []
        for pauli, angle in zip(self.__pauli_bits, self.__angles):
            multiple = angle / (np.pi / 2)
            if math.isclose(multiple, round(multiple)):
                new_pauli_bits.append(pauli)
                new_angles.append(angle)
        output = RotOps([pauli_bit_to_pauli_id(ele) for ele in new_pauli_bits])
        output.insert_angles(new_angles)
        return output

    def extract_non_clifford_part(self):
        new_pauli_bits = []
        new_angles = []
        for pauli, angle in zip(self.__pauli_bits, self.__angles):
            multiple = angle / (np.pi / 2)
            if not math.isclose(multiple, round(multiple)):
                new_pauli_bits.append(pauli)
                new_angles.append(angle)
        output = RotOps([pauli_bit_to_pauli_id(ele) for ele in new_pauli_bits])
        output.insert_angles(new_angles)
        return output

    def get_qubit_count(self):
        v = self.__pauli_bits[0][1:]
        return len(v)

    def __len__(self):
        return len(self.__pauli_bits)

    def __getitem__(self, position):
        return self.get_pauli_strings()[position], self.__angles[position]

    def __setitem__(self, index, value_lst):  # 値を代入したいときに使う
        new_value, new_angle = value_lst
        if isinstance(new_value, str):  # ["II", "XX", "II", "YY"]
            self.__pauli_bits[index] = pauli_string_to_pauli_bit(new_value)
        else:  # [[0,0], [1,1], [0,0], [2,2]]
            self.__pauli_bits[index] = pauli_id_to_pauli_bit(new_value)
        self.__angles[index] = new_angle

    def get_all(self):
        tmp = []
        pauli_ids = self.get_pauli_ids()
        for i in range(len(pauli_ids)):
            tmp.append((pauli_ids[i], self.__angles[i]))
        return tmp

    def insert_angles(self, angle_list: list[float]):
        assert len(angle_list) == len(
            self.__pauli_bits
        ), f"Pauli列の要素数({len(self.__pauli_bits)})と一致していません: {angle_list}"
        self.__angles = angle_list

    def insert_angles_from_sgn(
        self, sgn_list: list
    ):  # 符号だけで角度情報を入力できます
        assert len(sgn_list) == len(
            self.__pauli_bits
        ), f"Pauli列の要素数({len(self.__pauli_bits)})と一致していません: {sgn_list}"
        tmp = []
        for sgn in sgn_list:
            if sgn == 1:
                tmp.append(np.pi / 4)
            elif sgn == -1:
                tmp.append(-np.pi / 4)
            else:
                raise ValueError(f"Invalid input: {sgn}")
        self.__angles = tmp

    def reverse_all_angles(self):
        tmp = [-1 * val for val in self.__angles]
        self.__angles = tmp

    def duplicate(self):
        output = RotOps(self.get_pauli_strings().copy())
        angles = self.get_angles().copy()
        output.insert_angles(angles)
        return output

    def merge(self, another_rot_ops):
        left = self.get_pauli_bits()
        right = another_rot_ops.get_pauli_bits()
        self.__pauli_bits = left + right
        self.__angles = self.__angles + another_rot_ops.get_angles()

    # T layer に分割するアルゴリズムの実装(arXiv:2407.08695 Algorithm 1)
    def grouping(rot, target_number=-1):
        from mcr.litinski_compile import grouping_of_pauli_rotations

        data = grouping_of_pauli_rotations(rot)
        if target_number >= 0:
            for sublist in data:
                if target_number in sublist:
                    return sublist
        return grouping_of_pauli_rotations(rot)

    def extract(self, start: int, end: int):
        info = self.get_pauli_ids()[start:end].copy()
        angle_info = self.get_angles()[start:end].copy()
        output_rot_ops = RotOps(list(info))
        output_rot_ops.insert_angles(list(angle_info))
        return output_rot_ops

    def extract_from_positions(self, positions):
        info = self.get_pauli_ids().copy()
        angle_info = self.get_angles().copy()
        if isinstance(positions, list):
            objects, object_angles = [], []
            for pos in positions:
                objects.append(info[pos])
                object_angles.append(angle_info[pos])
        else:
            objects = [info[positions]]
            object_angles = [angle_info[positions]]
        output_rot_ops = RotOps(objects)
        output_rot_ops.insert_angles(object_angles)
        return output_rot_ops

    def get_qulacs_circuit(self):
        pauli_ids = self.get_pauli_ids()
        angles = [
            -1 * angle for angle in self.__angles
        ]  # qulacsの回転ゲートに-1をかける
        n = len(pauli_ids[0])
        gates = [
            PauliRotation([i for i in range(n)], pauli, angle)
            for pauli, angle in zip(pauli_ids, angles)
        ]

        circuit = QuantumCircuit(n)
        for gate in gates:
            circuit.add_gate(gate)
        return circuit

    def convert_to_clifford_t_circuit(self, gates_only=False, complement_id=True):
        # pauli_bitsの中に係数-1のものが含まれていないかを確認
        for pauli_bit in self.get_pauli_bits():
            if pauli_bit[0] != 0:
                raise ValueError(f"Invalid pauli_bit input: {pauli_bit}")

        nqubits = len(self.get_pauli_ids()[0])
        circuit = QuantumCircuit(nqubits)

        for gate in self.get_all():
            pauli_ids, angle = gate
            # Pauli演算子が入っているqubitのindexを取得
            qubit_indices = [i for i in range(nqubits) if pauli_ids[i] != 0]
            position = find_non_zero_index(pauli_ids)  # 最後の非ゼロのindex
            non_identity_pauli_ids = [p for p in pauli_ids if p != 0]

            # CNOTゲートが必要かを判断
            if len(qubit_indices) >= 2:  # CNOT必要
                apply_pauli_gates(
                    circuit, qubit_indices, non_identity_pauli_ids, right_side=False
                )

                # CNOTゲートを適用
                for idx in qubit_indices:
                    if idx != position:
                        circuit.add_gate(CNOT(idx, position))

                # 回転ゲートを適用
                apply_rotation_gate(circuit, angle, position)

                # CNOTゲートを逆順に適用
                for idx in reversed(qubit_indices):
                    if idx != position:
                        circuit.add_gate(CNOT(idx, position))

                # Pauliゲートを逆順に適用
                apply_pauli_gates(
                    circuit, qubit_indices, non_identity_pauli_ids, right_side=True
                )

            else:  # CNOT不要
                apply_pauli_gates(
                    circuit, qubit_indices, non_identity_pauli_ids, right_side=False
                )
                apply_rotation_gate(circuit, angle, position)
                apply_pauli_gates(
                    circuit, qubit_indices, non_identity_pauli_ids, right_side=True
                )
        if gates_only:
            return [circuit.get_gate(i) for i in range(circuit.get_gate_count())]
        else:
            if complement_id:
                return complement_identity(circuit)
            else:
                return circuit

    def draw(self):
        if self.get_qubit_count() < 10:
            c = self.convert_to_clifford_t_circuit()
            return circuit_drawer(c, "mpl")
        else:
            raise ValueError(f"Too many qubits to draw: {self.get_qubit_count()}")

    def save_qasm(self, file_name: str):
        from mcr.filesave import qulacs_to_qasm

        tmp = self.convert_to_clifford_t_circuit().copy()
        qulacs_to_qasm(file_name, tmp)

    # 後で文字列から計算される行列と値が一致するか確認する
    def get_matrix(self):
        circ = self.get_qulacs_circuit()
        mat = get_merged_matrix(circ)
        return mat

    def is_commute(self, position_1: int, position_2: int) -> bool:
        data = self.get_pauli_bits()
        tuple_bits1, tuple_bits2 = data[position_1][1:], data[position_2][1:]
        sgn = 1
        for i in range(len(tuple_bits1)):
            target = [tuple_bits1[i], tuple_bits2[i]]
            if target[0] != target[1] and (0, 0) not in target:  # anti-commute
                sgn *= -1
        if sgn == 1:
            return True
        return False

    def check_adjacent_commute(self):
        pauli_ids = self.get_pauli_ids()
        result = []
        for i in range(len(pauli_ids) - 1):
            tuple_id1, tuple_id2 = pauli_ids[i], pauli_ids[i + 1]
            sgn = 1
            for i in range(len(tuple_id1)):
                target = [tuple_id1[i], tuple_id2[i]]
                if target[0] != target[1] and 0 not in target:  # anti-commute
                    sgn *= -1
            if sgn == 1:
                result.append(True)
            else:
                result.append(False)
        return result

    def all_commutable_in_one_group(self):
        gate_count = self.get_gate_count()
        for i in range(gate_count):
            for j in range(i + 1, gate_count):
                if not self.is_commute(i, j):
                    return False
        return True

    def has_common_pauli_rotation(self):
        s = dict(Counter(self.get_pauli_ids()))
        values = list({key: value for key, value in s.items() if value >= 2}.keys())
        if len(values) > 0:
            return values
        else:
            return False

    def has_consecutive(self):
        lst = self.get_pauli_ids()
        for i in range(len(lst) - 1):
            if lst[i] == lst[i + 1]:
                return True
        return False

    def has_duplicates(self):
        lst = self.get_pauli_ids()
        if len(Counter([tuple(ele) for ele in lst])) < len(lst):
            return True
        else:
            return False

    def get_each_pauli_matrix(self, include_sgn: bool = False):
        matrices = []
        number_to_matrix = {
            0: np.eye(2),
            1: X(0).get_matrix(),
            2: Y(0).get_matrix(),
            3: Z(0).get_matrix(),
        }
        pauli_ids = self.get_pauli_ids()
        sgns = self.get_sgn_angles()
        for i in range(len(pauli_ids)):
            pauli_id = pauli_ids[i]
            tmp = []
            for num in pauli_id:
                tmp.append(number_to_matrix[num])
            mat = np.eye(1)
            for matrix in tmp:
                mat = np.kron(mat, matrix)
            matrices.append(mat)
        if include_sgn:
            data = [angle * mat for angle, mat in zip(sgns, matrices)]
            return data
        return matrices

    def is_swappable(self, another_rot_ops):
        circ1 = self.get_qulacs_circuit()
        circ2 = another_rot_ops.get_qulacs_circuit()
        nqubits = circ1.get_qubit_count()
        circ_before, circ_after = QuantumCircuit(nqubits), QuantumCircuit(nqubits)
        circ_before.merge_circuit(circ1)
        circ_before.merge_circuit(circ2)
        circ_after.merge_circuit(circ2)
        circ_after.merge_circuit(circ1)
        mat1 = get_merged_matrix(circ_before)
        mat2 = get_merged_matrix(circ_after)
        return are_unitaries_equivalent(mat1, mat2)

    # Apply the sign of the rotation axis to the rotation angle
    def apply_rot_axis_sgn_to_angles(self):
        original_pauli_bit_data = self.get_pauli_bits().copy()
        new_data = [
            pauli_bit_to_pauli_id(ele, with_coef=False)
            for ele in original_pauli_bit_data
        ]
        angle_sgn_update = []
        for pauli_bit in original_pauli_bit_data:
            sgn = pauli_bit[0]
            if sgn == 0:  # 係数1
                angle_sgn_update.append(1)
            elif sgn == 2:  # 係数-1
                angle_sgn_update.append(-1)
            else:  # 係数i, -iの場合
                raise ValueError(f"Invalid input: {sgn}")

        output = RotOps(new_data)
        angles = self.get_angles().copy()
        angles = [angle * sgn for angle, sgn in zip(angles, angle_sgn_update)]
        output.insert_angles(angles)
        return output
