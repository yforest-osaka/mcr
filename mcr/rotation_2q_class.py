from collections import Counter

import numpy as np
import pyzx as zx
import sympy as sp
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, H, PauliRotation, S, Sdag, T, Tdag, X, Y, Z
from qulacs_core import ClsOneQubitGate

from mcr.circuit_ops import get_merged_matrix
from mcr.clifford import complement_identity, is_clifford
from mcr.filesave import qulacs_to_pyzx

# 2qubit回転ゲートに関する様々な定義や操作を行うクラス
pauli_str_to_id_dict = {
    "II": (0, 0),
    "IX": (0, 1),
    "IY": (0, 2),
    "IZ": (0, 3),
    "XI": (1, 0),
    "XX": (1, 1),
    "XY": (1, 2),
    "XZ": (1, 3),
    "YI": (2, 0),
    "YX": (2, 1),
    "YY": (2, 2),
    "YZ": (2, 3),
    "ZI": (3, 0),
    "ZX": (3, 1),
    "ZY": (3, 2),
    "ZZ": (3, 3),
}

id_to_pauli_str_dict = {v: k for k, v in pauli_str_to_id_dict.items()}

true_dicts = {
    (0, 1): [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
    (0, 2): [(1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2)],
    (0, 3): [(1, 0), (1, 3), (2, 0), (2, 3), (3, 0), (3, 3)],
    (1, 0): [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
    (1, 1): [(0, 1), (1, 0), (2, 2), (2, 3), (3, 2), (3, 3)],
    (1, 2): [(0, 2), (1, 0), (2, 1), (2, 3), (3, 1), (3, 3)],
    (1, 3): [(0, 3), (1, 0), (2, 1), (2, 2), (3, 1), (3, 2)],
    (2, 0): [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3)],
    (2, 1): [(0, 1), (1, 2), (1, 3), (2, 0), (3, 2), (3, 3)],
    (2, 2): [(0, 2), (1, 1), (1, 3), (2, 0), (3, 1), (3, 3)],
    (2, 3): [(0, 3), (1, 1), (1, 2), (2, 0), (3, 1), (3, 2)],
    (3, 0): [(0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (3, 3)],
    (3, 1): [(0, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 0)],
    (3, 2): [(0, 2), (1, 1), (1, 3), (2, 1), (2, 3), (3, 0)],
    (3, 3): [(0, 3), (1, 1), (1, 2), (2, 1), (2, 2), (3, 0)],
}


def add_T_orTdag(seed: int, index: int) -> ClsOneQubitGate:
    """TゲートもしくはTdagゲートを返す

    Args:
        index (int): qubit_index
        seed (int, optional): Tなら1、Tdagなら-1(1以外)を指定する. Defaults to 1.

    Returns:
        ClsOneQubitGate: T or Tdag gate
    """
    if seed == 1:
        return T(index)
    else:
        return Tdag(index)


def multiply(string1, string2):
    pauli_product_dicts = {
        "II": [1, "I"],
        "IX": [1, "X"],
        "IY": [1, "Y"],
        "IZ": [1, "Z"],
        "XI": [1, "X"],
        "XX": [1, "I"],
        "XY": [1j, "Z"],
        "XZ": [-1j, "Y"],
        "YI": [1, "Y"],
        "YX": [-1j, "Z"],
        "YY": [1, "I"],
        "YZ": [1j, "X"],
        "ZI": [1, "Z"],
        "ZX": [1j, "Y"],
        "ZY": [-1j, "X"],
        "ZZ": [1, "I"],
    }
    assert len(string1) == len(string2)
    coefs = 1
    paulis = ""
    for i in range(len(string1)):
        pauli_left, pauli_right = string1[i], string2[i]
        coef, pauli = pauli_product_dicts[pauli_left + pauli_right]
        coefs *= coef
        paulis += pauli
    return coefs, paulis


def calculation(original_coef_dict, applying_gates):
    multiply_counter = 0
    results = {
        "II": 0,
        "IX": 0,
        "IY": 0,
        "IZ": 0,
        "XI": 0,
        "XX": 0,
        "XY": 0,
        "XZ": 0,
        "YI": 0,
        "YX": 0,
        "YY": 0,
        "YZ": 0,
        "ZI": 0,
        "ZX": 0,
        "ZY": 0,
        "ZZ": 0,
    }
    # for dict in [sin_dicts,cos_dicts]:
    non_zero_keys = {key: value for key, value in original_coef_dict.items() if value != 0}
    paulis = list(non_zero_keys.keys())  # 元々の
    coefs = list(non_zero_keys.values())

    for i, pauli in enumerate(paulis):
        for gate in applying_gates:
            original_coef = coefs[i]  # pauliから取り出す
            # print(gate)
            new_phase, new_paulis = multiply(pauli, gate[1])
            results[new_paulis] += original_coef * gate[0] * new_phase
            multiply_counter += 1
    # print(multiply_counter)
    return results


def swapping_4_4_matrix(matrix):
    assert matrix.shape[0] == 4
    SWAP_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return SWAP_MATRIX @ matrix @ SWAP_MATRIX


def tensor(pauli_id1, pauli_id2):
    x, y, z = X(0).get_matrix(), Y(0).get_matrix(), Z(0).get_matrix()
    string_to_matrix = {"I": np.eye(2), "X": x, "Y": y, "Z": z}
    return np.kron(string_to_matrix[pauli_id1], string_to_matrix[pauli_id2])


def gen_matrix_form_dicts(dictionary):
    COS = sp.Symbol("c")
    SIN = sp.Symbol("s")
    matrix = np.zeros([4, 4])
    for ele in dictionary.keys():
        coef = complex(dictionary[ele].subs([(SIN, np.sin(np.pi / 8)), (COS, np.cos(np.pi / 8))]).evalf())
        assert tensor(ele[0], ele[1]).shape == (4, 4)
        term = coef * tensor(ele[0], ele[1])
        matrix = matrix + term
    return matrix


def rotation_to_gates(pauli_id, phase):
    COS = sp.Symbol("c")
    SIN = sp.Symbol("s")
    if phase == np.pi / 4:
        return [[COS, "II"], [-1j * SIN, pauli_id]]
    elif phase == -np.pi / 4:
        return [[COS, "II"], [1j * SIN, pauli_id]]
    else:
        raise ValueError(f"Invalid input: {pauli_id, phase}")


def synthesis(gates_element_list):
    initial_coefs = {
        "II": 0,
        "IX": 0,
        "IY": 0,
        "IZ": 0,
        "XI": 0,
        "XX": 0,
        "XY": 0,
        "XZ": 0,
        "YI": 0,
        "YX": 0,
        "YY": 0,
        "YZ": 0,
        "ZI": 0,
        "ZX": 0,
        "ZY": 0,
        "ZZ": 0,
    }
    coefs = initial_coefs.copy()

    initial_data1, initial_data2 = gates_element_list[0]

    coefs[initial_data1[1]] = initial_data1[0]
    coefs[initial_data2[1]] = initial_data2[0]

    tmp = coefs.copy()
    for gate in gates_element_list[1:]:
        tmp = calculation(tmp, gate)
    value = tmp.copy()

    non_zero_keys = {key: value for key, value in value.items() if value != 0}
    return non_zero_keys


class Rot2QOps:

    def __init__(self, gate_sequence: list[str] | list[list[int]] | list[tuple[int]]) -> None:
        input_element = gate_sequence[0]
        if (
            # isinstance(input_element[0], str) and len(input_element) == 2 # mistake
            isinstance(input_element[0], str)
            and isinstance(input_element, tuple)
        ):  # [('II',angle1), ('XX',angle2)), ('II',angle3)), ('YY',angle4))]
            pauli_strings = []
            angles = []
            for elem in gate_sequence:
                pauli, angle = elem
                pauli_strings.append(pauli)
                angles.append(angle)
            self.__pauli_strings = pauli_strings
            self.__pauli_ids = [pauli_str_to_id_dict[pauli] for pauli in pauli_strings]
            self.__angles = angles
        elif isinstance(input_element[0], list) or isinstance(
            input_element[0], tuple
        ):  # [([0,0],angle1), ([1,1],angle2)), ([0,0],angle3)), ([2,2],angle4))]
            pauli_ids = []
            angles = []
            for elem in gate_sequence:
                pauli, angle = elem
                pauli_ids.append(tuple(pauli))
                angles.append(angle)
            self.__pauli_ids = pauli_ids
            self.__pauli_strings = [id_to_pauli_str_dict[tuple(pauli)] for pauli in pauli_ids]
            self.__angles = angles

        elif isinstance(input_element, str):  # ["II", "XX", "II", "YY"]
            self.__pauli_strings = gate_sequence
            self.__pauli_ids = [pauli_str_to_id_dict[pauli] for pauli in gate_sequence]
            self.__angles = [0 for _ in range(len(self.__pauli_strings))]
        else:
            if not isinstance(input_element, list) and not isinstance(
                input_element, tuple
            ):  # [[0,0], [1,1], [0,0], [2,2]]
                raise ValueError(f"Invalid input: {gate_sequence}")
            self.__pauli_ids = [tuple(pauli) for pauli in gate_sequence]
            self.__pauli_strings = [id_to_pauli_str_dict[tuple(pauli)] for pauli in gate_sequence]
            self.__angles = [0 for _ in range(len(self.__pauli_strings))]

    def get_pauli_ids(self):
        return self.__pauli_ids

    def get_angles(self):
        return self.__angles

    def get_sgn_angles(self):
        return np.sign(self.__angles)

    def get_pauli_strings(self):
        return self.__pauli_strings

    def get_all(self):
        tmp = []
        for i in range(len(self.__pauli_ids)):
            tmp.append((self.__pauli_ids[i], self.__angles[i]))
        return tmp

    def insert_angles(self, angle_list: list[float]):
        assert len(angle_list) == len(
            self.__pauli_strings
        ), f"Pauli列の要素数({len(self.__pauli_strings)})と一致していません: {angle_list}"
        self.__angles = angle_list

    def insert_angles_from_sgn(self, sgn_list: list):  # 符号だけで角度情報を入力できます
        assert len(sgn_list) == len(
            self.__pauli_strings
        ), f"Pauli列の要素数({len(self.__pauli_strings)})と一致していません: {sgn_list}"
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

    def get_qulacs_circuit(self):
        pauli_ids = self.__pauli_ids
        angles = [-1 * angle for angle in self.__angles]  # qulacsの回転ゲートに-1をかける
        gates = [PauliRotation([0, 1], pauli, angle) for pauli, angle in zip(pauli_ids, angles)]

        circuit = QuantumCircuit(2)
        for gate in gates:
            circuit.add_gate(gate)
        return circuit

    def convert_to_clifford_t_circuit(self):
        circuit = QuantumCircuit(2)
        # pauli_idはそのまま&回転角を反転させる
        for gate in self.get_all():
            # pauli_ids = list(reversed(gate[0]))
            pauli_ids = gate[0]
            # angle = -1 * gate[1]
            angle = gate[1]  # 回転角は反転させない！
            t_id = 0
            if angle > 0:  # t gate
                t_id = 1
            # pauli_idに0が入っているか否かで場合わけ
            if 0 not in pauli_ids:  # 要CNOT
                for idx, pauli_id in zip([0, 1], pauli_ids):
                    if pauli_id == 1:
                        circuit.add_gate(H(idx))
                    elif pauli_id == 2:
                        circuit.add_gate(Sdag(idx))
                        circuit.add_gate(H(idx))
                circuit.add_gate(CNOT(0, 1))
                circuit.add_gate(add_T_orTdag(t_id, 1))
                circuit.add_gate(CNOT(0, 1))
                for idx, pauli_id in zip([0, 1], pauli_ids):
                    if pauli_id == 1:
                        circuit.add_gate(H(idx))
                    elif pauli_id == 2:
                        circuit.add_gate(H(idx))
                        circuit.add_gate(S(idx))
            else:  # CNOT不要
                for idx, pauli_id in zip([0, 1], pauli_ids):
                    if pauli_id > 0:
                        position = idx
                    if pauli_id == 1:
                        circuit.add_gate(H(idx))
                    elif pauli_id == 2:
                        circuit.add_gate(Sdag(idx))
                        circuit.add_gate(H(idx))
                circuit.add_gate(add_T_orTdag(t_id, position))
                for idx, pauli_id in zip([0, 1], pauli_ids):
                    if pauli_id == 1:
                        circuit.add_gate(H(idx))
                    elif pauli_id == 2:
                        circuit.add_gate(H(idx))
                        circuit.add_gate(S(idx))
        return complement_identity(circuit)

    # 後で文字列から計算される行列と値が一致するか確認する
    def get_matrix(self):
        circ = self.get_qulacs_circuit()
        mat = get_merged_matrix(circ)
        return mat

    def is_commute(self, position_1: int, position_2: int) -> bool:
        pauli_ids = self.get_pauli_ids()
        # print(f"pauli_ids: {pauli_ids}")
        upper, lower = [pauli_ids[position_1][0], pauli_ids[position_2][0]], [
            pauli_ids[position_1][1],
            pauli_ids[position_2][1],
        ]
        sgn = 1
        for ele in [upper, lower]:
            if ele[0] != ele[1] and 0 not in ele:  # anti-commute
                sgn *= -1
        if sgn == 1:
            return True
        return False

    def check_commute(self):
        pauli_ids = self.get_pauli_ids()
        # print(f"pauli_ids: {pauli_ids}")
        result = []
        for i in range(len(pauli_ids) - 1):
            upper, lower = [pauli_ids[i][0], pauli_ids[i + 1][0]], [pauli_ids[i][1], pauli_ids[i + 1][1]]
            sgn = 1
            for ele in [upper, lower]:
                if ele[0] != ele[1] and 0 not in ele:  # anti-commute
                    sgn *= -1
            if sgn == 1:
                result.append(True)
            else:
                result.append(False)
        return result

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

    def can_locally_optimize(self):
        pauli_ids = self.get_pauli_ids()
        # print(f'Input: {pauli_ids}')
        target_pauli_ids = self.has_common_pauli_rotation()  # 共通の回転軸を持っているか
        if target_pauli_ids == False:
            return False
        else:
            if self.has_consecutive():
                return True
            else:
                for target_pauli in target_pauli_ids:
                    # print(f'target_pauli: {target_pauli}')
                    commutable_pauli_ids = true_dicts[target_pauli]
                    # print(f'commutable_pauli_ids: {commutable_pauli_ids}')
                    positions = [
                        i for i, ele in enumerate(pauli_ids) if ele == target_pauli
                    ]  # 今考えているpauliがいる場所
                    for j in range(len(positions) - 1):
                        start, end = positions[j], positions[j + 1]
                        # print(f'start-end: {start}, {end}')
                        checker = [pauli_ids[i] in commutable_pauli_ids for i in range(start + 1, end)]
                        # print(checker)
                        if all(checker) == True:
                            # print(f'start-end: {start}, {end}')
                            return True
                return False

    def get_pyzx_optimized_graph(self, show_graph: bool = False):
        circuit_qulacs = self.convert_to_clifford_t_circuit()
        # Cliffordか判定
        c_zx = qulacs_to_pyzx(circuit_qulacs)
        g = c_zx.to_graph()
        zx.full_reduce(g)
        if show_graph:
            g.normalize()
            zx.draw(g)
        return g.copy()

    # この2qubit回転ゲートが非自明なClifford回路を生成するかどうかを判定する
    def is_non_trivial_clifford(self, show_graph: bool = False):

        circuit_qulacs = self.convert_to_clifford_t_circuit()
        # Cliffordか判定
        mat = get_merged_matrix(circuit_qulacs)
        if not is_clifford(mat):
            return 0
        else:
            c_zx = qulacs_to_pyzx(circuit_qulacs)
            g = c_zx.to_graph()
            zx.full_reduce(g)
            if show_graph:
                g.normalize()
                zx.draw(g)
            c_aft = zx.extract_circuit(g.copy())
            if c_aft.tcount() > 0:
                # print('発見！')
                # return data
                return 2
            else:
                return 1

    def get_rot_calculation(self, decimal: int = 3, get_matrix: bool = False, show_coef: bool = True):
        COS = sp.Symbol("c")
        SIN = sp.Symbol("s")
        # data = self.get_all()
        data = []
        for i in range(len(self.__pauli_strings)):
            data.append((self.__pauli_strings[i], self.__angles[i]))
        gates = list(
            reversed([rotation_to_gates(ele[0], ele[1]) for ele in data])
        )  # 量子回路と表現を一致させるために順番を逆にする
        non_zero_keys = synthesis(gates)
        # 項別に表示
        if show_coef:
            for ele in non_zero_keys.keys():
                term = non_zero_keys[ele]
                coef = complex(term.subs([(SIN, np.sin(np.pi / 8)), (COS, np.cos(np.pi / 8))]).evalf())
                print(ele, np.round(coef, decimal))
        if get_matrix:
            return gen_matrix_form_dicts(non_zero_keys)
