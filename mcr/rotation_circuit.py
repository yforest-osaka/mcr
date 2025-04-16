# 基本ライブラリ
import collections
import copy
from typing import List

# プロット
import matplotlib.pyplot as plt
import stim

# Qulacs関連
from qulacs import QuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
from qulacs.converter import convert_qulacs_circuit_to_QASM
from qulacsvis import circuit_drawer as draw


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
        Parameters:
        gate_id (tuple[int]): A tuple representing the gate identifier.
        gate (stim.PauliString): The gate to be added, which must be an instance of a qulacs gate.
        Raises:
        ValueError: If the gate is not an instance of a qulacs gate.
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
            self.index_distribution[f"{index}"] += 1  # index_distributionの更新

    def __getitem__(self, position: int) -> tuple:
        """circuit[position]をした時の定義

        Args:
            position (int): 取得したいゲートの位置

        Returns:
            tuple: (gate_id, gate)
        """
        return self.__gate_list[position]  # listのposition番目を取ってくる

    def __len__(self) -> int:
        """len()をつけた時の定義(削除されるべきゲートも含むことに注意！確認用)

        Returns:
            int: ゲートの数(削除されるべきゲートも含むことに注意！)
        """
        return len(self.__gate_list)

    def list(self) -> list:
        """list()をつけた時の定義(削除されるべきゲートも含むことに注意！確認用)

        Returns:
            list: ゲートのリスト(削除されるべきゲートも含むことに注意！)
        """
        return self.__gate_list

    def get_qubit_count(self) -> int:
        """qubitの数を取得するメソッド

        Returns:
            int: qubitの数
        """
        return self.__n_qubit

    def get_index_distribution(self, show_graph: bool = False) -> collections.Counter:
        """qubitの入っている箇所の分布を取得するメソッド

        Args:
            show_graph (bool, optional): グラフを表示するかどうか. Defaults to False.

        Returns:
            collections.Counter: ゲートの入っている箇所の分布(Counter)
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
        )  # ゲートの入っていないようなqubitが存在してもworkする

    def get_all(
        self,
        target_id: tuple | bool = False,
        only_id: bool = False,
        include_position: bool = False,
    ) -> List:
        """全てのゲート列を取得するメソッド

        Args:
            target_id (tuple | NoneType, optional): 指定したIDの入ったリストに属するゲートのみ取得する. Defaults to None.
            only_id (bool, optional): 指定したIDに一致するゲートのみ取得する. Defaults to False.
            include_position (bool, optional): 条件を満たすゲートの位置も取得する. Defaults to False.

        Returns:
            list: ゲートのリスト
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
        # if include_position:
        #     return datalist, positions
        # else:
        #     return datalist
        return sorted(datalist, key=lambda x: x[0])

    def get_all_ids(self) -> List[tuple[int]]:
        """全てのゲートのIDを取得するメソッド

        Returns:
            list[tuple[int]]: ゲートのIDのリスト
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
        """ゲート削除を行うメソッド

        Args:
            position (int): 削除したいゲートの位置

        Raises:
            ValueError: すでに削除されているゲートを削除しようとした場合
        """
        if position in self.del_nums:
            raise ValueError(
                f"The position {position} is already included in del_nums!:{self.del_nums}"
            )
        self.del_nums.add(position)
        # index_distributionを更新(削除されるものを反映する)
        indices = self.__gate_list[position][1].pauli_indices()
        for index in indices:
            self.index_distribution[f"{index}"] -= 1  # index_distributionの更新

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
        """生成したゲートのリストを本来の新しい順番に並べ替えるメソッド

        Returns:
            List: 並べ替えたゲートのリスト
        """
        if only_gates:
            return [ele[1] for ele in sorted(self.get_all())]
        else:
            return sorted(self.get_all())

    def to_rot_ops(self) -> List:
        new_rot_ops = []
        sgns = []
        data = self.sort_gate_sequence()
        for value in data:
            string = str(value[1])
            if string[0] == "+":
                sgns.append(1)
            else:
                sgns.append(-1)
            pauli_string = string[1:]
            pauli_string = pauli_string.replace("_", "I")
            new_rot_ops.append(pauli_string)
        rot = RotOps(new_rot_ops)
        rot.insert_angles_from_sgn(sgns)
        return rot

    def set_circuit(self) -> QuantumCircuit:
        """Qulacsの回路にセットするメソッド

        Returns:
            QuantumCircuit: Qulacsの回路
        """

        rot_ops = self.to_rot_ops()
        circ = rot_ops.convert_to_clifford_t_circuit(complement_id=True)
        return circ

    def draw(self):
        """量子回路を描画するメソッド"""
        qulacs_circuit = self.set_circuit()
        return draw(qulacs_circuit, "mpl")

    def get_inversed_gates(self) -> List:
        """インバースの回路を取得するメソッド

        Returns:
            List: インバースの回路
        """
        qulacs_circuit = self.set_circuit()
        inverse = qulacs_circuit.get_inverse()
        gates = []
        for i in range(inverse.get_gate_count()):
            gates.append(inverse.get_gate(i))
        return gates

    def get_gate(self, position: int):
        """指定した位置のゲートを取得するメソッド

        Args:
            position (int): 取得したいゲートの位置

        Returns:
            QulacsGate: Qulacsのゲート
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
        """ゲートの数を取得するメソッド

        Returns:
            int: ゲートの数
        """
        t = len(self.sort_gate_sequence())
        return t

    def is_equivalent(self, another_circuit: "MyQuantumProgram") -> float:
        """2つの量子回路が等価かどうかを判定するメソッド"""
        from mcr.circuit_ops import equivalence_check_via_mqt_qcec

        qulacs_circuit1 = self.set_circuit()
        qulacs_circuit2 = another_circuit.set_circuit()
        exclude_zx_checker = True
        if equivalence_check_via_mqt_qcec(
            qulacs_circuit1, qulacs_circuit2, exclude_zx_checker, show_log=False
        ):
            return True
        else:
            from mcr.circuit_ops import are_unitaries_equivalent, get_merged_matrix

            m1 = get_merged_matrix(qulacs_circuit1)
            m2 = get_merged_matrix(qulacs_circuit2)
            return are_unitaries_equivalent(m1, m2)

    def duplicate(self) -> "PauliRotationSequence":
        """量子回路を複製するメソッド

        Returns:
            PauliRotationSequence: 複製された量子回路
        """
        tmp_circ = PauliRotationSequence(self.__n_qubit)
        tmp_circ.__gate_list = copy.deepcopy(self.__gate_list)  # gate_listのコピー
        tmp_circ.index_distribution = copy.deepcopy(
            self.index_distribution
        )  # index_distributionのcopy
        tmp_circ.del_nums = self.del_nums.copy()  # del_numsのコピー
        return tmp_circ

    def merge(self, block_size: int) -> List:
        """指定した量子回路をブロックサイズでマージするメソッド

        Args:
            block_size (int): ブロックサイズ

        Returns:
            List: マージされた量子回路
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
        """量子回路をQASM形式で保存するメソッド

        Args:
            filename (str): 保存するファイル名
        """
        qulacs_circ = self.set_circuit()
        string = convert_qulacs_circuit_to_QASM(qulacs_circ)
        with open(filename, mode="w") as f:
            f.write("\n".join(string))


# N qubit Pauli Rotation Class
import math
from collections import Counter

import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, H, PauliRotation, S, Sdag, T, Tdag, X, Y, Z
from qulacsvis import circuit_drawer

from mcr.clifford import complement_identity


def pauli_bit_to_pauli_id(pauli_bit, with_coef=True):
    sgn_dict = {0: 1, 1: 1j, 2: -1, 3: -1j}
    pauli_bit_to_id_dict = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    if with_coef:
        return sgn_dict[pauli_bit[0]], tuple(
            pauli_bit_to_id_dict[i] for i in pauli_bit[1:]
        )
    else:
        return tuple(pauli_bit_to_id_dict[i] for i in pauli_bit[1:])


def pauli_string_to_pauli_bit(pauli_string):  # 1個のPauli_stringを与える
    pauli_str_to_bit_dict = {
        "I": (0, 0),
        "X": (0, 1),
        "Y": (1, 1),
        "Z": (1, 0),
        "_": (0, 0),
    }
    if isinstance(
        pauli_string, tuple
    ):  # タプルの内部で係数が与えられている場合 (-1, 'XX')
        sgn_dict = {1: 0, 1j: 1, -1: 2, -1j: 3}
        sgn_bit = sgn_dict[pauli_string[0]]
        pauli_string = pauli_string[1]
    else:
        sgn_bit = 0  # 係数1
    if pauli_string[:2] == "+i":  # 先頭が文字列の+iの場合(stimでの利用を想定)
        sgn_bit = 1
        pauli_string = pauli_string[2:]
    elif pauli_string[:2] == "-i":  # 先頭が文字列の-iの場合(stimでの利用を想定)
        sgn_bit = 3
        pauli_string = pauli_string[2:]
    elif pauli_string[0] == "-":  # 先頭が文字列の-の場合(stimでの利用を想定)
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

    def merge(self, another_rot_ops):
        left = self.get_pauli_bits()
        right = another_rot_ops.get_pauli_bits()
        self.__pauli_bits = left + right
        self.__angles = self.__angles + another_rot_ops.get_angles()

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
