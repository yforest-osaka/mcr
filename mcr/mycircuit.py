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

# Qiskit関連
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector

# Qulacs関連
from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
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

# Qulacs可視化
from qulacsvis import circuit_drawer

# エラー設定
np.seterr(all="ignore")


def get_inverse_gate(
    parametric_gate: ParametricPauliRotation,
) -> ParametricPauliRotation:
    """ParametricPauliRotationを引数として、その回転角を逆にしたゲートを取得する

    Args:
        parametric_gate (ParametricPauliRotation): 逆にしたいParametricPauliRotation

    Returns:
        ParametricPauliRotation: 逆にしたParametricPauliRotation
    """
    target_indices, paulis, parameter = get_gate_info(parametric_gate)
    pauli_dict = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    pauli_id = []
    for pauli in paulis:
        pauli_id.append(pauli_dict[pauli])
    return ParametricPauliRotation(target_indices, pauli_id, -1 * parameter)


class MyQuantumProgram:
    def __init__(self, n_qubit: int) -> None:
        """Initializes a quantum program with a specified number of qubits.

        Args:
            n_qubit (int): The number of qubits in the quantum program.
        """
        self.__gate_list = []
        self.del_nums = set()
        self.index_distribution = {f"{i}": 0 for i in range(n_qubit)}
        self.__n_qubit: int = n_qubit

    def add_gate(self, gate_id: tuple[int], gate) -> None:
        """ゲートを追加するメソッド

        Args:
            gate_id (tuple[int]): ゲートのID
            gate : Qulacsのゲート

        Raises:
            ValueError: ゲートがQulacsゲートになっているかを確認
        """
        if not self.__is_qulacs_gate(gate):  # ゲートがQulacsゲートになっているかを確認
            raise ValueError("The gate must be an instance of a qulacs gate.")
        self.__gate_list.append((gate_id, gate))
        indices = gate.get_target_index_list()
        for index in indices:
            self.index_distribution[f"{index}"] += 1  # index_distributionの更新

    def __is_qulacs_gate(self, gate: ClsOneQubitGate) -> bool:
        """Checks if the specified gate is a qulacs gate.

        Args:
            gate (ClsOneQubitGate): The gate to check.

        Returns:
            bool: True if the gate is a qulacs gate, False otherwise.
        """

        return "qulacs" in gate.__module__ and "Gate" in str(gate.__class__)

    def add_identity_gateset(self, arg_depth):
        qubit_index = list(range(self.__n_qubit))
        i = 0
        Identity = np.eye(4)
        for _ in range(arg_depth):
            random.shuffle(qubit_index)
            for k in range(self.__n_qubit // 2):
                targets = sorted([qubit_index[2 * k], qubit_index[2 * k + 1]])
                self.add_gate((i,), DenseMatrix(targets, Identity))
                i += 1

    def __add_random_qc_gateset(self, depth: int) -> None:
        """ランダムにRandomUnitaryゲートをdepth分だけ追加するメソッド

        Args:
            depth (int): ランダムに追加するRandomUnitary回路のdepth
        """
        qubit_index = list(range(self.__n_qubit))
        i = 0
        for _ in range(depth):
            random.shuffle(qubit_index)
            for k in range(self.__n_qubit // 2):
                targets = sorted([qubit_index[2 * k], qubit_index[2 * k + 1]])
                self.add_gate((i,), RandomUnitary(targets))
                i += 1

    def __add_clifford_t_qc_gateset(self, depth: int, angles: list[float]) -> None:
        """ランダムにClifford+Tゲートをdepth分だけ追加するメソッド

        Args:
            depth (int): ランダムに追加するClifford+T回路のdepth
            angles (list): 回転角のリスト
        """
        qubit_index = list(range(self.__n_qubit))
        i = 0
        for _ in range(depth):
            random.shuffle(qubit_index)
            for k in range(self.__n_qubit // 2):
                # angles = [-np.pi/4, -np.pi/2, np.pi/4, np.pi/2]
                targets = sorted([qubit_index[2 * k], qubit_index[2 * k + 1]])
                while True:
                    # 重複を許して2つランダムに選ぶ(Identityは除く)
                    pauli_ids = random.choices([1, 2, 3], k=2)
                    # 結果が[0, 0]であればやり直す
                    if pauli_ids != [0, 0]:
                        break
                self.add_gate(
                    (i,),
                    ParametricPauliRotation(targets, pauli_ids, random.choice(angles)),
                )
                i += 1

    def __add_rzz_qc_gateset(self, depth: int, angles: list[float]) -> None:
        """ランダムにRZZゲートをdepth分だけ追加するメソッド

        Args:
            depth (int): ランダムに追加するRZZ回路のdepth
            angles (list[float]): 回転角のリスト
        """
        print("RZZ")
        qubit_index = list(range(self.__n_qubit))
        i = 0
        for _ in range(depth):
            random.shuffle(qubit_index)
            for k in range(self.__n_qubit // 2):
                # angles = [-np.pi/4, -np.pi/2, np.pi/4, np.pi/2]
                targets = sorted([qubit_index[2 * k], qubit_index[2 * k + 1]])
                pauli_ids = [3, 3]
                self.add_gate(
                    (i,),
                    ParametricPauliRotation(targets, pauli_ids, random.choice(angles)),
                )
                i += 10

    def add_qc_gateset(
        self,
        arg_depth: int,
        arg_circuit_type: str = "RandomUnitary",
        arg_angles: list[float] = [np.pi / 4, -np.pi / 4],
    ) -> None:
        """ランダム量子回路を配置するメソッド

        Args:
            arg_depth (int): 量子回路の深さ
            arg_circuit_type (str, optional): 量子回路の種類. Defaults to "RandomUnitary".
            arg_angles (list[float], optional): 回転角のリスト. Defaults to [0, np.pi, -np.pi].

        Raises:
            ValueError: arg_circuit_typeが不正な値の場合
        """
        if arg_circuit_type not in ["RandomUnitary", "Clifford_t", "Rzz"]:
            raise ValueError(f"the argument is incorrect: {arg_circuit_type}")
        if arg_circuit_type == "RandomUnitary":
            self.__add_random_qc_gateset(arg_depth)
        elif arg_circuit_type == "Clifford_t":
            self.__add_clifford_t_qc_gateset(arg_depth, arg_angles)
        elif arg_circuit_type == "Rzz":
            self.__add_rzz_qc_gateset(arg_depth, arg_angles)

    def __getitem__(self, position: int) -> tuple:
        """circuit[position]をした時の定義

        Args:
            position (int): 取得したいゲートの位置

        Returns:
            tuple: (gate_id, gate)
        """
        return self.__gate_list[position]  # listのposition番目を取ってくる

    def __setitem__(self, position: int, gate) -> None:
        """[ ]をした時の定義.新しいゲートを代入したい時に使う value -> ((tuple),QuantumGate)

        Args:
            position (int): 代入したい位置
            gate (_type_): 代入したいゲート

        Raises:
            ValueError: ゲートがQulacsゲートになっているかを確認
        """
        # (削除されるべきゲートも含むことに注意！確認用)
        if not self.__is_qulacs_gate(gate):  # ゲートのtypeが合っているかを確認
            raise ValueError("The gate must be an instance of a qulacs gate.")
        # index_distributionを更新(削除されるものを反映する)
        indices = self.__gate_list[position][1].get_target_index_list()
        for index in indices:
            self.index_distribution[f"{index}"] -= 1  # index_distributionの更新
        # 新しく加わるものを反映
        indices = gate.get_target_index_list()
        for index in indices:
            self.index_distribution[f"{index}"] += 1  # index_distributionの更新
        self.__gate_list[position] = (
            self.__gate_list[position][0],
            gate,
        )  # gate_idは継承する

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
            for index in gate.get_target_index_list():
                results.append(index)
        data = collections.Counter(results)
        if show_graph:
            plt.bar(list(data.keys()), list(data.values()), align="center")
            plt.xlabel("index")
            plt.ylabel("gate_count")
        return collections.Counter(
            self.index_distribution
        )  # ゲートの入っていないようなqubitが存在してもworkする

    def get_index_distribution_in_nqubits(
        self, show_graph: bool = False, max_block_size: int = 3
    ) -> collections.Counter:
        """max_block_sizeのブロックサイズ単位でのゲートの分布を取得するメソッド

        Args:
            show_graph (bool, optional): グラフを表示するかどうか. Defaults to False.
            max_block_size (int, optional): ブロックサイズ. Defaults to 3.

        Returns:
            collections.Counter: ゲートの入っている箇所の分布(Counter)
        """
        results = []
        tmp = self.set_circuit()
        print("depth: ", tmp.calculate_depth())
        tmp2 = tmp.copy()
        QCO().optimize(tmp2, max_block_size)
        print("depth: ", tmp2.calculate_depth())
        for i in range(tmp2.get_gate_count()):
            index = tmp2.get_gate(i).get_target_index_list()
            for num in index:
                results.append(num)
        data = collections.Counter(results)
        if show_graph:
            plt.bar(list(data.keys()), list(data.values()), align="center")
            plt.xlabel("index")
            plt.ylabel("gate_count")
        return data, tmp2

    def get_all_gatetypes(self) -> List[tuple[int]]:
        """どの種類のゲートが何個あるかを取得するメソッド

        Returns:
            list[tuple[int]]: ゲートの種類とその数
        """

        results = []
        for gate in self.__gate_list:
            results.append(gate[1].get_name())
        c = collections.Counter(results)
        return c.most_common()

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
        if include_position:
            return datalist, positions
        else:
            return datalist

    def get_all_ids(self) -> List[tuple[int]]:
        """全てのゲートのIDを取得するメソッド

        Returns:
            list[tuple[int]]: ゲートのIDのリスト
        """
        return [
            data[0] for i, data in enumerate(self.__gate_list) if i not in self.del_nums
        ]

    def draw(self, cut_position: int = False):
        """量子回路の可視化を行うメソッド"""
        display_circ = QuantumCircuit(self.__n_qubit)
        data = self.sort_gate_sequence()
        if cut_position:
            data = data[:cut_position]
        for gate in data:
            display_circ.add_gate(gate[1])
        return circuit_drawer(display_circ, "mpl")

    def get_state(self, initial_state: int | str) -> QuantumState:
        """指定した量子状態から量子回路を適用した後の状態を取得するメソッド

        Args:
            initial_state (int | str): 量子状態

        Returns:
            QuantumState: 量子回路を適用した後の状態
        """
        state = QuantumState(self.__n_qubit)
        if initial_state:
            state.set_computational_basis(int(initial_state))
        else:
            state.set_zero_state()
        tmp_circ = self.set_circuit()
        tmp_circ.update_quantum_state(state)
        return state

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
        indices = self.__gate_list[position][1].get_target_index_list()
        for index in indices:
            self.index_distribution[f"{index}"] -= 1  # index_distributionの更新

    def sort_gate_sequence(self) -> List:
        """生成したゲートのリストを本来の新しい順番に並べ替えるメソッド

        Returns:
            List: 並べ替えたゲートのリスト
        """
        return sorted(self.get_all())

    def get_existing_parameter_values(
        self,
    ) -> List:
        """ParametricPauliRotationはQuantumCircuitに入れるとなぜかパラメータ情報が消失するので回転角のデータを持っておく"""
        parameters = []
        data = self.sort_gate_sequence()
        for elem in data:
            if elem[1].get_name() == "ParametricPauliRotation":
                parameters.append(elem[1].get_parameter_value())
        return parameters

    def set_circuit(self, show_log: bool = False) -> QuantumCircuit:
        """Qulacsの回路にセットするメソッド

        Args:
            show_log (bool, optional): ログを表示するかどうか. Defaults to True.

        Returns:
            QuantumCircuit: Qulacsの回路
        """
        circ = QuantumCircuit(self.__n_qubit)
        data = self.sort_gate_sequence()
        for elem in data:
            circ.add_gate(elem[1])
            if show_log:
                print(f"added: {data[0]}")
        return circ

    def save_circuit(self, filename: str) -> None:
        """量子回路を保存するメソッド

        Args:
            filename (str): 保存するファイル名
        """
        n = self.get_qubit_count()
        with open(f"./circuit_data_{n}/{filename}.pickle", "wb") as f:
            gate = self.get_gate_count()
            circ = self.set_circuit()
            pickle.dump(({"nqubits": n, "total_gates": gate}, circ), f)

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

    def get_gate_count(self) -> int:
        """ゲートの数を取得するメソッド

        Returns:
            int: ゲートの数
        """
        t = len(self.sort_gate_sequence())
        qulacs_circuit = self.set_circuit()
        assert t == qulacs_circuit.get_gate_count()
        return t

    def get_depth(self) -> int:
        """量子回路の深さを取得するメソッド"""
        qulacs_circuit = self.set_circuit()
        return qulacs_circuit.calculate_depth()

    def is_equivalent(self, another_circuit: "MyQuantumProgram") -> float:
        """2つの量子回路が等価かどうかを判定するメソッド"""
        ket = self.get_state()
        bra = another_circuit.get_state()

        fidelity = abs(inner_product(bra, ket))
        if fidelity < 0.95:
            raise ValueError(f"Fidelity is incorrect: {fidelity}")
        return fidelity

    def calc_equivalence(self, another_circuit: "MyQuantumProgram"):
        """2つの量子回路が等価かどうかを判定するメソッド"""
        ket = self.get_state()
        bra = another_circuit.get_state()

        fidelity = abs(inner_product(bra, ket))
        return fidelity

    def equivalence_full_check(self, another_circuit: "MyQuantumProgram"):
        """2つの量子回路が等価かどうかを(2^n個の基底全てについて調べ)判定するメソッド"""
        for i in range(2 ** (self.__n_qubit)):
            ket = self.get_state(i)
            bra = another_circuit.get_state(i)
            fidelity = abs(inner_product(bra, ket))
            if fidelity < 0.95 or fidelity > 1.05:
                raise ValueError(f"Fidelity is incorrect: {fidelity}")
        return fidelity

    # def is_identity(self, accuracy=1e-3):
    #     data = []
    #     for i in range(2 ** (self.__n_qubit)):
    #         bra = QuantumState(self.__n_qubit)
    #         bra.set_computational_basis(int(i))
    #         ket = self.get_state(i)
    #         phase = np.angle(inner_product(bra, ket))
    #         data.append(phase)

    #     if max(data) - min(data) > accuracy:
    #         raise ValueError(f"Not Identity: {data}")
    #     return True

    def duplicate(self) -> "MyQuantumProgram":
        """量子回路を複製するメソッド

        Returns:
            MyQuantumProgram: 複製された量子回路
        """
        tmp_circ = MyQuantumProgram(self.__n_qubit)
        tmp_circ.__gate_list = self.__gate_list.copy()  # gate_listのコピー
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


class GateFactory:
    """ゲートを生成するクラス(swapなどの特殊なゲートを生成する)"""

    def __init__(self, gate):
        self.gate = gate
        self.target = gate.get_target_index_list()

    def __and__(self, other):
        return list(set(self.target) & set(other.target))

    def __or__(self, other):
        return list(set(self.target) | set(other.target))

    def gen_qubit_map(self, another_gate):
        """2つのゲートのtargetのindexを対応させるための辞書を生成するメソッド"""
        set1 = sorted(self.__or__(another_gate))
        set2 = [i for i in range(len(set1))]
        return {set1[i]: set2[i] for i in range(len(set1))}

    def complement_identity(self, another_gate):
        """2つのゲートのtargetが重複していない部分に適切にIdentityゲートを挿入してゲートのサイズを揃えるメソッド"""
        n = len(self.__or__(another_gate))
        # print(self.target,another_gate.target)
        tmp_circ = QuantumCircuit(n)
        swap_map = self.gen_qubit_map(another_gate)
        # gate1を変換
        converted_target = [swap_map[number] for number in self.target]
        complemented_index = [i for i in range(n) if i not in converted_target]
        tmp_circ.add_gate(DenseMatrix(converted_target, self.gate.get_matrix()))
        for j in complemented_index:
            tmp_circ.add_gate(Identity(j))
        QCO().optimize(tmp_circ, n)
        g1 = tmp_circ.get_gate(0)

        tmp_circ = QuantumCircuit(n)
        # gate2を変換
        converted_target = [swap_map[number] for number in another_gate.target]
        complemented_index = [i for i in range(n) if i not in converted_target]
        tmp_circ.add_gate(DenseMatrix(converted_target, another_gate.gate.get_matrix()))
        for j in complemented_index:
            tmp_circ.add_gate(Identity(j))
        QCO().optimize(tmp_circ, n)
        g2 = tmp_circ.get_gate(0)
        return g1, g2

    def recalculate_matrix(
        self, another_gate, option: str
    ) -> tuple[list[int], np.ndarray, list[int], np.ndarray]:
        """2つの行列の入れ替えに伴い生成される新しい行列を計算するメソッド"""
        gate_left, gate_right = self.complement_identity(another_gate)
        # test_circuit1 = QuantumCircuit(3)
        mat_A = gate_left.get_matrix()
        mat_B = gate_right.get_matrix()
        if option == "keep_left":
            left_matrix = np.conjugate(mat_A).T @ mat_B @ mat_A
            # np.save('left.npy',left_matrix)
            right_matrix = mat_A
        elif option == "keep_right":
            right_matrix = mat_B @ mat_A @ np.conjugate(mat_B).T
            left_matrix = mat_B
        else:
            raise ValueError(f"The argument option is incorrect: {option}")
        left_index = gate_right.get_target_index_list()
        right_index = gate_left.get_target_index_list()

        assert sorted(left_index) == sorted(right_index)  # おそらく常に[0,1,2]になる？
        return left_index, left_matrix, right_index, right_matrix

    def swap(self, another_gate, option: str, run_kak: bool, qubit_count=None) -> list:
        """2つのゲートを入れ替えるメソッド"""
        # 重複するindexがあるかないかを判定する機構(無ければ単に入れ替えるだけ)
        if (
            len(set(self.target) & set(another_gate.target)) == 0
        ):  # おそらくこの条件にかかることはない
            print("No change")
            return [another_gate.gate, self.gate]
        idx1, mat1, idx2, mat2 = self.recalculate_matrix(another_gate, option)
        swap_map = self.gen_qubit_map(another_gate)
        inverse_swap_map = {v: k for k, v in swap_map.items()}
        # gate1を元の形へ
        # print(self.gen_qubit_map(another_gate))
        # print('inverse: ',inverse_swap_map)
        # print('idx1: ',idx1)
        converted_target1 = [inverse_swap_map[number] for number in idx1]
        converted_target2 = [inverse_swap_map[number] for number in idx2]

        if (
            np.allclose(
                mat1.dot(np.conj(mat1.T)), np.eye(mat1.shape[0]), rtol=1e-05, atol=1e-08
            )
            == False
        ):
            raise ValueError(f"Matrix is not unitary: {mat1}")
        # mat1_load = np.load('tmp_new.npy')
        # accuracy = 4 # 精度を設定することができる
        # mat1 = np.round(mat1,accuracy)
        if option == "keep_left":
            tmp = my_3q_decomp(mat1, converted_target1, opt_run_kak=run_kak)
        elif option == "keep_right":
            tmp = my_3q_decomp(mat2, converted_target2, opt_run_kak=run_kak)

        # 生成したゲートの内 indexが逆順のものは直す
        gates = []
        if option == "keep_right":
            gates.append(another_gate.gate)

        for i in range(tmp.get_gate_count()):
            gate = tmp.get_gate(i)
            tg = gate.get_target_index_list()
            if sorted(tg) == tg:
                gates.append(gate)
            else:
                print("SWAP!!!")
                tg = list(reversed(tg))
                swap = SWAP(0, 1).get_matrix()
                gates.append(DenseMatrix(tg, swap @ gate.get_matrix() @ swap))

        if option == "keep_left":
            gates.append(self.gate)
        return gates

    def swap_without_3q_gate(
        self, another_gate, option, remove_edge_gate: bool = False
    ):
        """3量子ビットゲートの分解を使わずに2つのゲートを入れ替えるメソッド(only available for Clifford + T)"""
        # 重複するindexがあるかないかを判定する機構(無ければ単に入れ替えるだけ)
        if (
            len(set(self.target) & set(another_gate.target)) == 0
        ):  # おそらくこの条件にかかることはない
            print("No change")
            return [another_gate.gate, self.gate]

        gates = []
        if option == "keep_left":
            gates.append(self.gate)
            gates.append(another_gate.gate)
            gates.append(get_inverse_gate(self.gate))
            if not remove_edge_gate:
                gates.append(self.gate)
        elif option == "keep_right":
            gates.append(another_gate.gate)
            gates.append(get_inverse_gate(another_gate.gate))
            gates.append(self.gate)
            gates.append(another_gate.gate)
        return gates


def get_gate_info(gate, with_parameter: bool = True) -> tuple:
    """ゲートの情報を取得するメソッド"""
    text = gate.to_string()
    results_pauli_ids = []
    positions = [m.start() for m in re.finditer("commute", text)]
    for position in positions:
        sequence = text[position : position + 15]
        if "X" in sequence:
            results_pauli_ids.append("X")
        elif "Y" in sequence:
            results_pauli_ids.append("Y")
        elif "Z" in sequence:
            results_pauli_ids.append("Z")
        else:
            results_pauli_ids.append("I")
    if with_parameter:
        return (
            gate.get_target_index_list(),
            results_pauli_ids,
            gate.get_parameter_value(),
        )
    else:
        return gate.get_target_index_list(), results_pauli_ids


def get_residue_from_parameter(value) -> int:
    """パラメータをmod8した数を取得するメソッド

    Args:
        value (float): パラメータ

    Returns:
        int: mod8した数
    """
    return np.round(value * 4 / np.pi) % 8


def get_gates_from_parameter(residue: int, target_index: int) -> list:
    """パラメータからゲートを生成するメソッド"""
    gate_dicts = {
        0: [Identity(target_index)],
        1: [T(target_index)],
        2: [S(target_index)],
        3: [S(target_index), T(target_index)],
        4: [Z(target_index)],
        5: [Sdag(target_index), Tdag(target_index)],
        6: [Sdag(target_index)],
        7: [Tdag(target_index)],
    }
    return gate_dicts[residue]


def two_qubit_pauli_gate_converter_of_qulacs(
    qulacs_circuit: QuantumCircuit, parameters: list[float]
) -> QuantumCircuit:
    """Qulacsの回路をPauliゲートに変換するメソッド"""
    circuit_out = QuantumCircuit(qulacs_circuit.get_qubit_count())
    counter = 0
    for n in range(qulacs_circuit.get_gate_count()):
        gate = qulacs_circuit.get_gate(n)
        if gate.get_name() != "ParametricPauliRotation":
            circuit_out.add_gate(gate)

        else:  # ParametricPauliRotation
            indices, paulis = get_gate_info(gate, with_parameter=False)
            parameter = parameters[counter]
            counter += 1
            parameter = -1 * parameter  # 回転角を反転

            # 前のレイヤー
            control, target = indices
            for idx, pauli in zip(indices, paulis):
                if pauli == "X":
                    circuit_out.add_gate(H(idx))
                elif pauli == "Y":
                    circuit_out.add_gate(Sdag(idx))
                    circuit_out.add_gate(H(idx))
                else:
                    pass
            # Identityを含む場合は不要
            if "I" not in paulis:
                circuit_out.add_gate(CNOT(control, target))

            # 中央(パラメータ依存)
            residue = get_residue_from_parameter(parameter)
            if "I" not in paulis:
                for i, gate in enumerate(get_gates_from_parameter(residue, target)):
                    circuit_out.add_gate(gate)
            else:
                if paulis[0] != "I":
                    for i, gate in enumerate(
                        get_gates_from_parameter(residue, control)
                    ):
                        circuit_out.add_gate(gate)
                else:
                    for i, gate in enumerate(get_gates_from_parameter(residue, target)):
                        circuit_out.add_gate(gate)
            # Identityを含む場合は不要
            if "I" not in paulis:
                circuit_out.add_gate(CNOT(control, target))

            # 後のレイヤー
            for idx, pauli in zip(indices, paulis):
                if pauli == "X":
                    circuit_out.add_gate(H(idx))
                elif pauli == "Y":
                    circuit_out.add_gate(H(idx))
                    circuit_out.add_gate(S(idx))
                else:
                    pass
    assert len(parameters) == counter
    return circuit_out
