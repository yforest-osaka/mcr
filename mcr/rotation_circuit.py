# 基本ライブラリ
import collections
import copy
import pickle
import random
import re
from typing import List, Type, Union

# プロット
import matplotlib.pyplot as plt
import numpy as np
import stim

# Qiskit関連
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector

# Qulacs関連
from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
from qulacs.converter import convert_qulacs_circuit_to_QASM
from qulacs.gate import (  # type: ignore
    CNOT,
    SWAP,
    DenseMatrix,
    H,
    Identity,
    ParametricPauliRotation,
    RandomUnitary,
    S,
    Sdag,
    T,
    Tdag,
    Z,
)
from qulacs.state import inner_product  # type: ignore
from qulacs_core import ClsOneQubitGate
from qulacsvis import circuit_drawer as draw

from mcr.circuit_ops import assign_gate_id_of_generated_gates
from mcr.rot_class import RotOps


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
