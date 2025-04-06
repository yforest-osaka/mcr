import pickle
from itertools import product

import numpy as np
from qulacs import QuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
from qulacs.converter import convert_QASM_to_qulacs_circuit
from qulacs.gate import Identity, T, Tdag  # type: ignore
from qulacsvis import circuit_drawer

from mcr.clifford import complement_identity, ctrl_gates_to_dem


def generate_gate_combinations(circuit_id: int, qubit_count: int = 2) -> QuantumCircuit:
    """I, T, Tdagを並べた回路において、指定したIDの回路を生成する

    Args:
        circuit_id (int): 生成したい回路のID
        qubit_count (int, optional): 量子ビット数. Defaults to 2.

    Returns:
        QuantumCircuit: 生成した回路
    """
    circ = QuantumCircuit(qubit_count)
    gate_classes = [Identity, T, Tdag]

    combinations = list(product(gate_classes, repeat=qubit_count))
    generator_dicts = {
        i + 1: [gate_class(j) for j, gate_class in enumerate(combo)] for i, combo in enumerate(combinations)
    }
    for gate in generator_dicts[circuit_id]:
        circ.add_gate(gate)
    return circ


class TwoQubitClT:  # 2qubitのClifford+T回路を表すクラス
    def __init__(self, filepath: str) -> None:
        """TwoQubitClTクラスの初期化

        Args:
            filepath (str): ファイルパス
        """
        if "pickle" in filepath:
            self.filepath = filepath
            with open(filepath, mode="br") as f:
                data = pickle.load(f)
            path_str = data[0]
            data[0] = f"../data/{path_str}"  # パスを微修正
            self.circuit_file, self.left_id, self.right_id = data
        else:
            self.filepath = filepath
            self.circuit_file = filepath
            self.left_id, self.right_id = 1, 1
            # raise ValueError('Nothing')

    def get_circuit(self, add_left: bool, add_right: bool) -> QuantumCircuit:
        """回路を取得する

        Args:
            add_left (bool): 左側パートの回路を追加するかどうか
            add_right (bool): 右側パートの回路を追加するかどうか

        Returns:
            QuantumCircuit: 生成した回路
        """
        with open(self.circuit_file, mode="r") as f:
            circ_qasm = f.read()
        c = convert_QASM_to_qulacs_circuit(circ_qasm.splitlines())
        if add_left and add_right:
            left = generate_gate_combinations(self.left_id)
            right = generate_gate_combinations(self.right_id)
            left.merge_circuit(c)
            left.merge_circuit(right)
            return left
        if add_left and not add_right:  # leftだけ追加
            left = generate_gate_combinations(self.left_id)
            left.merge_circuit(c)
            return left
        elif not add_left and add_right:  # rightだけ追加
            right = generate_gate_combinations(self.right_id)
            c.merge_circuit(right)
            return c
        else:
            assert add_left == False
            assert add_right == False
            return c

    def get_inverse_circuit(self, add_left: bool, add_right: bool) -> QuantumCircuit:
        """逆回路を取得する

        Args:
            add_left (bool): 左側パートの回路を追加するかどうか
            add_right (bool): 右側パートの回路を追加するかどうか

        Returns:
            QuantumCircuit: 生成した逆回路
        """
        with open(self.circuit_file, mode="r") as f:
            circ_qasm = f.read()
        c = convert_QASM_to_qulacs_circuit(circ_qasm.splitlines())
        if add_left and add_right:
            left = generate_gate_combinations(self.left_id)
            right = generate_gate_combinations(self.right_id)
            left.merge_circuit(c)
            left.merge_circuit(right)
            return left.get_inverse()
        if add_left and not add_right:  # leftだけ追加
            left = generate_gate_combinations(self.left_id)
            left.merge_circuit(c)
            return left.get_inverse()
        elif not add_left and add_right:  # rightだけ追加
            right = generate_gate_combinations(self.right_id)
            c.merge_circuit(right)
            return c.get_inverse()
        else:
            assert add_left == False
            assert add_right == False
            return c.get_inverse()

    def get_matrix(self, add_left: bool = True, add_right: bool = True, decimals: int = False) -> np.ndarray:
        """回路の行列を取得する

        Args:
            add_left (bool, optional): 左側パートの回路を追加するかどうか. Defaults to True.
            add_right (bool, optional): 右側パートの回路を追加するかどうか. Defaults to True.
            decimals (int, optional): 出力する行列の有効桁数. Defaults to False.

        Returns:
            np.ndarray: 生成した行列
        """
        circuit = self.get_circuit(add_left, add_right)
        circuit = complement_identity(circuit)
        circuit = ctrl_gates_to_dem(circuit)
        QCO().optimize(circuit, 2)
        mat = circuit.get_gate(0).get_matrix()
        if decimals:
            return np.round(mat, decimals)
        else:
            return mat

    def draw(self, add_left: bool = True, add_right: bool = True):
        circuit = self.get_circuit(add_left, add_right)
        return circuit_drawer(circuit, "mpl")

    def text_draw(self, add_left: bool = True, add_right: bool = True):
        circuit = self.get_circuit(add_left, add_right)
        return circuit_drawer(circuit)

    def get_gates_from_id(self, opt: str) -> list:
        """IDからゲートを取得する

        Args:
            opt (str): オプションの値。"right"または"left"のいずれかを指定する。

        Returns:
            list: ゲートのリスト。指定されたオプションに応じて、右側のゲートまたは左側のゲートのリストを返す。
        """
        gate_classes = [Identity, T, Tdag]

        combinations = list(product(gate_classes, repeat=2))
        generator_dicts = {
            i + 1: [gate_class(j) for j, gate_class in enumerate(combo)] for i, combo in enumerate(combinations)
        }
        if opt == "right":
            return generator_dicts[self.right_id]
        else:
            assert opt == "left"
            return generator_dicts[self.left_id]

    def is_matrix_product(self) -> bool:
        """行列積の形になっているかどうかを判定する

        Returns:
            bool: 行列積の形になっている場合はTrue、そうでない場合はFalse
        """
        circuit = self.get_circuit(add_left=True, add_right=True)
        QCO().optimize(circuit, 1)
        if circuit.get_gate_count() == circuit.get_qubit_count():
            return True
        else:
            return False

    def get_cnot_count(self) -> int:
        """CNOTの数をカウントする

        Returns:
            int: CNOTの数
        """
        circuit = self.get_circuit(add_left=True, add_right=True)
        # cnot数をカウントする
        counter = 0
        for i in range(circuit.get_gate_count()):
            if "CNOT" in circuit.get_gate(i).get_name():
                counter += 1
        return counter

    def is_commute(self) -> bool:
        """可換かどうかを判定する

        Returns:
            bool: 可換の場合はTrue、そうでない場合はFalse
        """
        circuit_center_only = self.get_circuit(add_left=False, add_right=False)
        commute_flag = False
        # left
        t_circ_l = QuantumCircuit(2)
        gates = self.get_gates_from_id(opt="left")
        for gate in gates:
            t_circ_l.add_gate(gate)
        c1 = t_circ_l.copy()
        c1.merge_circuit(circuit_center_only)
        QCO().optimize(c1, 2)
        m1 = c1.get_gate(0).get_matrix()

        c2 = circuit_center_only.copy()
        c2.merge_circuit(t_circ_l)
        QCO().optimize(c2, 2)
        m2 = c2.get_gate(0).get_matrix()
        if np.allclose(m1, m2):
            commute_flag = True

        # right
        t_circ_r = QuantumCircuit(2)
        gates = self.get_gates_from_id(opt="right")
        for gate in gates:
            t_circ_r.add_gate(gate)

        c3 = t_circ_r.copy()
        c3.merge_circuit(circuit_center_only)
        QCO().optimize(c3, 2)
        m3 = c3.get_gate(0).get_matrix()

        c4 = circuit_center_only.copy()
        c4.merge_circuit(t_circ_r)
        QCO().optimize(c4, 2)
        m4 = c4.get_gate(0).get_matrix()
        if np.allclose(m3, m4):
            commute_flag = True
        return commute_flag

    def has_trivial_swap(self) -> bool:
        """SWAPが含まれていて単品のCNOTを含んでいないような回路かどうかを判定する

        Returns:
            bool: SWAPが含まれていて単品のCNOTを含んでいない場合はTrue、そうでない場合はFalse
        """
        swap_flag = False
        if self.get_cnot_count() != 3:
            return swap_flag
        else:
            indices = str()
            circuit = self.get_circuit(add_left=False, add_right=False)
            for i in range(circuit.get_gate_count()):
                gate = circuit.get_gate(i)
                if "CNOT" in gate.get_name():
                    indices += str(gate.get_control_index_list()[0])
                    indices += str(gate.get_target_index_list()[0])
                else:
                    indices += "n"
            if "011001" in indices or "100110" in indices:
                swap_flag = True
            return swap_flag

    def get_t_count(self) -> int:
        """要素に含まれているtの数をカウントする(2個or4個)

        Raises:
            ValueError: idに問題がある場合

        Returns:
            int: Tゲートの数
        """
        value = 0
        for num in [self.left_id, self.right_id]:
            if num in [2, 3, 4, 7]:
                value += 1
            elif num in [5, 6, 8, 9]:
                value += 2
            else:
                raise ValueError(f"idに問題があります: {num}")
        return value
