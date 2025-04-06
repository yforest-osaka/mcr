# 必要な外部ライブラリ
import math
import re
import uuid

import numpy as np
import pandas as pd

# Qulacs関連のインポート
from qulacs import QuantumCircuit

# pytket関連のインポート
# from pytket.circuit import Unitary3qBox
# from pytket.passes import DecomposeBoxes
# from pytket.qasm import circuit_to_qasm_str  # type: ignore
# from pytket.transform import Transform
# from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
from qulacs.gate import CZ, DenseMatrix, H, RotY, RotZ  # type: ignore
from qulacs.state import inner_product  # type: ignore

# Visualizationツール
from qulacsvis import circuit_drawer as draw
from scipy.linalg import cossin, schur

from mcr.filesave import qasm_to_qulacs, string_to_qasm

# 量子回路の非最適化関連
from mcr.kak import kak_decomposition

# from pytket import Circuit as TketCircuit  # type: ignore


# エラー処理の設定
np.seterr(all="ignore")


YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
CNOT_MATRIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ_MATRIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
SWAP_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def to_df(mat: np.ndarray) -> pd.DataFrame:
    """Converts a numpy array to a pandas DataFrame.

    Args:
        mat (np.ndarray): 行列

    Returns:
        pd.DataFrame: 行列をDataFrameに変換したもの
    """
    return pd.DataFrame(mat)


def my_multiplexed_angles(data_theta: list) -> np.ndarray:
    value = (
        np.array(
            [
                (data_theta[0] + data_theta[1] + data_theta[2] + data_theta[3]),
                (data_theta[0] + data_theta[1] - data_theta[2] - data_theta[3]),
                (data_theta[0] - data_theta[1] - data_theta[2] + data_theta[3]),
                (data_theta[0] - data_theta[1] + data_theta[2] - data_theta[3]),
            ]
        )
        / 4
    )
    return value


def my_cs_to_ops(theta: float, idxs: list) -> list:
    new_theta = my_multiplexed_angles(theta * 2)
    # print(new_theta/np.pi)
    return [
        RotY(idxs[2], new_theta[0]),
        DenseMatrix([idxs[1], idxs[2]], CZ_MATRIX),
        RotY(idxs[2], new_theta[1]),
        DenseMatrix([idxs[0], idxs[2]], CZ_MATRIX),
        RotY(idxs[2], new_theta[2]),
        DenseMatrix([idxs[1], idxs[2]], CZ_MATRIX),
        RotY(idxs[2], new_theta[3]),
    ]


def my_middle_multiplexor_to_ops(eigvals: np.ndarray, idxs: list) -> list:
    params = np.real(np.log(np.sqrt(eigvals)) * 1j * 2)
    new_theta_mid = my_multiplexed_angles(params)
    # print(new_theta_mid/np.pi)
    return [
        RotZ(idxs[2], new_theta_mid[0]),
        DenseMatrix([idxs[2], idxs[1]], CNOT_MATRIX),
        RotZ(idxs[2], new_theta_mid[1]),
        DenseMatrix([idxs[2], idxs[0]], CNOT_MATRIX),
        RotZ(idxs[2], new_theta_mid[2]),
        DenseMatrix([idxs[2], idxs[1]], CNOT_MATRIX),
        RotZ(idxs[2], new_theta_mid[3]),
        DenseMatrix([idxs[2], idxs[0]], CNOT_MATRIX),
    ]


def to_special(matrix: np.ndarray) -> np.ndarray:
    return matrix * (np.linalg.det(matrix) ** (-1 / len(matrix)))


def _gamma(matrix: np.ndarray) -> np.ndarray:
    return matrix @ YY @ matrix.T @ YY


def extract_right_diag(unitary_matrix: np.ndarray) -> np.ndarray:
    t = _gamma(to_special(unitary_matrix).T).diagonal()
    k = np.real(t[0] + t[3] - t[1] - t[2])
    psi = np.arctan2(np.imag(np.sum(t)), k)
    f = np.exp(1j * psi)
    return np.diag([1, f, f, 1])


def num_cnots_required(u: np.ndarray, atol: float = 1e-8) -> int:
    g = _gamma(to_special(u))
    # see Fadeev-LeVerrier formula
    a3 = -np.trace(g)
    # no need to check a2 = 6, as a3 = +-4 only happens if the eigenvalues are
    # either all +1 or -1, which unambiguously implies that a2 = 6
    if np.abs(a3 - 4) < atol or np.abs(a3 + 4) < atol:
        return 0
    # see Fadeev-LeVerrier formula
    a2 = (a3 * a3 - np.trace(g @ g)) / 2
    if np.abs(a3) < atol and np.abs(a2 - 2) < atol:
        return 1
    if np.abs(a3.imag) < atol:
        return 2
    return 3


def my_two_qubit_matrix_to_diagonal_and_cz_operations(mat: np.ndarray) -> tuple:
    if num_cnots_required(mat) == 3:
        # print('ここが呼ばれる')
        right_diag = extract_right_diag(mat)
        two_cnot_unitary = mat @ right_diag  # two_cnot_unitaryはkak分解される4*4行列。そのまま代入でOK！
        # print('check!!')
        # print(pd.DataFrame(two_cnot_unitary))
        # my_func_is_unitary(two_cnot_unitary)
        return right_diag.conj().T, two_cnot_unitary
    else:
        # print('関数終わります')
        return np.eye(4), mat


def my_unitary_eig(matrix: np.ndarray):  # シュール分解
    R, V = schur(matrix, output="complex")
    return R.diagonal(), V


def my_two_qubit_multiplexor_to_ops(
    u1: np.ndarray, u2: np.ndarray, idxs: list, shift_left: bool = True, diagonal: bool = None, run_kak: bool = False
):
    ops = []
    u1u2 = u1 @ u2.conj().T
    # eigvals, v = np.linalg.eig(u1u2) #4*4行列の固有値, 固有ベクトルを並べた行列
    eigvals, v = my_unitary_eig(u1u2)
    d = np.diag(np.sqrt(eigvals))
    w = d @ v.conj().T @ u2
    # print('w is: ',w)
    circuit_u1u2_mid = my_middle_multiplexor_to_ops(eigvals, idxs)
    if diagonal is not None:
        v = diagonal @ v
    # print('v is: ',v)
    d_v, circuit_u1u2_r = my_two_qubit_matrix_to_diagonal_and_cz_operations(v)  # q1,q2のみ!
    w = d_v @ w

    if shift_left:
        d_w, circuit_u1u2_l = my_two_qubit_matrix_to_diagonal_and_cz_operations(w)
    else:
        d_w = None
        circuit_u1u2_l = w
    index = idxs[:-1]
    if run_kak:
        # kak分解を実行して追加
        elems = kak_decomposition(circuit_u1u2_l, index)
        for elem in elems:
            ops.append(elem[1])
    else:
        ops.append(DenseMatrix(index, circuit_u1u2_l))
    ops.extend(circuit_u1u2_mid)
    if run_kak:
        # kak分解を実行して追加
        elems = kak_decomposition(circuit_u1u2_r, index)
        for elem in elems:
            ops.append(elem[1])
    else:
        ops.append(DenseMatrix(index, circuit_u1u2_r))
    return d_w, ops


def my_3q_decomp(matrix: np.ndarray, index: int, opt_run_kak: bool) -> QuantumCircuit:
    (u1, u2), theta, (v1h, v2h) = cossin(matrix, separate=True, p=4, q=4)
    u2 = u2 @ np.diag([1, -1, 1, -1])  # なんかこれが要るらしい
    gates_center = my_cs_to_ops(theta, index)  # cs
    # print('u1,u2 applies')
    d_ud, ud_ops = my_two_qubit_multiplexor_to_ops(
        u1, u2, index, shift_left=True, run_kak=opt_run_kak
    )  # 4*4matrix_u1&u2
    # print('v1h,v2h applies')
    _, vdh_ops = my_two_qubit_multiplexor_to_ops(
        v1h, v2h, index, shift_left=False, diagonal=d_ud, run_kak=opt_run_kak
    )  # 4*4matrix_v1&v2

    circuit = QuantumCircuit(max(index) + 1)
    for gate in vdh_ops:
        circuit.add_gate(gate)
    for gate in gates_center:
        circuit.add_gate(gate)
    for gate in ud_ops:
        circuit.add_gate(gate)
    QCO().optimize(circuit, 2)  # ここでマージする
    # 生成したゲートの内 indexが逆順のものは直す
    circuit_out = QuantumCircuit(max(index) + 1)
    for i in range(circuit.get_gate_count()):
        gate = circuit.get_gate(i)
        tg = gate.get_target_index_list()
        if sorted(tg) == tg:
            circuit_out.add_gate(gate)
        else:
            tg = list(reversed(tg))
            circuit_out.add_gate(DenseMatrix(tg, SWAP_MATRIX @ gate.get_matrix() @ SWAP_MATRIX))
    return circuit_out


def my_func_is_unitary(mat1: np.ndarray) -> None:
    if np.allclose(mat1.dot(np.conj(mat1.T)), np.eye(mat1.shape[0]), rtol=1e-05, atol=1e-08) == False:
        raise ValueError(f"Matrix is not unitary: {mat1}")


def get_qasm_data_from_parameter(param_number: int, qubit_index: int):
    """指定のパラメータを入力して指定のqubit_indexを代入したqasmファイルを出力する関数を作る

    Args:
        param_number (int): パラメータ番号
        qubit_index (int): qubitのindex

    Raises:
        ValueError: Not Found: {param_number}

    Returns:
        str: qasmファイルの文字列
    """
    tg_path = f"./data/sk_data_5_optimized/{param_number}.qasm"
    # if tg_path in files:
    try:  # ここにtg_pathから文字列を読み込むコマンド
        with open(tg_path, mode="r") as f:
            data = f.read()
        data = data.replace("q[0]", f"q[{qubit_index}]")
        return data
    except:
        raise ValueError(f"Not Found: {param_number}")


def gen_cl_t_gates(index: int, phase_val: float) -> list:
    """qubitのindexと回転角の値から、クリフォードゲートとTゲートの列を生成する

    Args:
        index (int): qubitのindex
        phase_val (float): 回転角

    Returns:
        list: ゲートのリスト
    """
    dicts = {
        0.0: [],
        0.25: [f"t q[{index}]"],
        0.5: [f"s q[{index}]"],
        0.75: [f"s q[{index}]", f"t q[{index}]"],
        1.0: [f"s q[{index}]", f"s q[{index}]"],
        1.25: [f"sdg q[{index}]", f"tdg q[{index}]"],
        1.5: [f"sdg q[{index}]"],
        1.75: [f"tdg q[{index}]"],
    }
    if phase_val in dicts:
        return dicts[phase_val]
    else:
        return False


def phase_converter(string, keep_angle=False):
    """rx, rzのparamを持ったqasmをclifford+tに変換する

    Args:
        string (str): qasmの文字列
        keep_angle (bool, optional): 角度を保持するかどうか. Defaults to False.

    Returns:
        str: 変換後のqasmの文字列
    """
    new_string = []
    initial_gates = string.split(";\n")
    new_counter = 0
    for ele in initial_gates:
        if "*pi" in ele:
            match_rz = re.search(r"\(([-\d\.]+)\*pi\)", ele)  # 負の角にも対応
            match_q = re.search(r"q\[(\d+)\]", ele)
            phase = match_rz.group(1)
            index = match_q.group(1)
            # phase = math.fmod(float(phase), 2.0)
            phase = float(phase) % 2.0

            gates = []
            if "rx" in ele:  # Rx回転
                gates.append(f"h q[{index}]")
            if keep_angle:
                angle = -phase * np.pi
                elements = [f"rz({angle}) q[{index}]"]
            else:
                elements = gen_cl_t_gates(index, phase)
            if elements != False:
                gates.extend(elements)
            else:  # 無限小数角を持ったパラメータ
                # ここにファイルパスを入力して指定のqubit_indexを代入したqasmファイルを出力する関数を作る
                new_counter += 1
                sk_gates = get_qasm_data_from_parameter(phase, qubit_index=index).split(";\n")
                for value in sk_gates:
                    if "OPENQASM" not in value and "include" not in value and "qreg" not in value:
                        gates.append(value)  # solovay-kitaevしたものを取得！

            if "rx" in ele:  # Rx回転
                gates.append(f"h q[{index}]")
            for val in gates:
                new_string.append(val)
        else:
            new_string.append(ele)
    # 最後に繋げる
    tmp = ";\n".join(new_string)
    # if new_counter>0:
    #     print('SK applied!')
    string = tmp.replace(";\n;\n", ";\n")
    return string


# def three_qubit_decomp_via_tket(mat, index, nqubits) -> QuantumCircuit:  # for solovay-kitaev(0705追加)
#     """Pytketを使って3qubitのユニタリ行列を分解する"""
#     c_tk = TketCircuit(nqubits)

#     reversed_indices = list(reversed(index))
#     c_tk.add_gate(Unitary3qBox(mat), reversed_indices)
#     DecomposeBoxes().apply(c_tk)
#     Transform.RebaseToPyZX().apply(c_tk)
#     string = phase_converter(circuit_to_qasm_str(c_tk))
#     unique_id = uuid.uuid4()
#     filepath = f"tmp/tmp_{unique_id}.qasm"
#     string_to_qasm(filepath, string, join_option=False)
#     circuit_out = qasm_to_qulacs(filepath)
#     return circuit_out


def cnot_to_cz(qc):
    """cnotのqasmをczのqasmに変換する"""
    if type(qc) == str:  # 文字列(qasm)の場合
        output_string = []
        initial_gates = qc.split(";\n")
        for ele in initial_gates:
            if "cx" in ele:
                # print(ele)
                # match_q = re.search(r'q\[(\d+)\]', ele)
                match_q = re.findall(r"\d+", ele)
                # index = match_q.group()
                index = [int(num) for num in match_q]
                # print(index)
                output_string += [f"h q[{index[1]}]", f"cz q[{index[0]}], q[{index[1]}]", f"h q[{index[1]}]"]
            else:
                output_string.append(ele)
        # 2回続くアダマールはここで取り除く
        had_counter_0 = []
        had_counter_1 = []
        remove_gate_number = set()
        for i, ele in enumerate(output_string):
            if "cz" in ele:  # reset
                had_counter_0, had_counter_1 = [], []
            elif "q[0]" in ele:  # 他のゲートが入った時もresetしないといけない！！！
                if "h" in ele:
                    had_counter_0.append(i)
                    if len(had_counter_0) == 2:
                        remove_gate_number.add(had_counter_0[0])
                        remove_gate_number.add(had_counter_0[1])
                        had_counter_0 = []
                else:
                    had_counter_0 = []
            elif "q[1]" in ele:
                if "h" in ele:
                    had_counter_1.append(i)
                    if len(had_counter_1) == 2:
                        remove_gate_number.add(had_counter_1[0])
                        remove_gate_number.add(had_counter_1[1])
                        had_counter_1 = []
                else:
                    had_counter_1 = []
        new_string = []
        for i, ele in enumerate(output_string):
            if i not in remove_gate_number:
                new_string.append(ele)
        # 最後に繋げる
        tmp = ";\n".join(new_string)
        string = tmp.replace(";\n;\n", ";\n")
        return string
    else:  # qulacsの場合
        n = qc.get_qubit_count()
        tmp_gates = []  # CNOT -> CZ
        for i in range(qc.get_gate_count()):
            gate = qc.get_gate(i)
            if gate.get_name() == "CNOT":
                ctrl = gate.get_control_index_list()[0]
                tg = gate.get_target_index_list()[0]
                tmp_gates.append(H(tg))
                tmp_gates.append(CZ(ctrl, tg))
                tmp_gates.append(H(tg))
            else:
                tmp_gates.append(gate)
        # 2回続くアダマールはここで取り除く
        had_counter_0 = []
        had_counter_1 = []
        remove_gate_number = set()
        for i, ele in enumerate(tmp_gates):
            tg = ele.get_target_index_list()[0]
            gate_name = ele.get_name()
            if gate_name == "CZ":  # reset
                had_counter_0, had_counter_1 = [], []
            elif tg == 0:  # 他のゲートが入った時もresetしないといけない！！！
                if gate_name == "H":
                    had_counter_0.append(i)
                    if len(had_counter_0) == 2:
                        remove_gate_number.add(had_counter_0[0])
                        remove_gate_number.add(had_counter_0[1])
                        had_counter_0 = []
                else:
                    had_counter_0 = []
            elif tg == 1:
                if gate_name == "H":
                    had_counter_1.append(i)
                    if len(had_counter_1) == 2:
                        remove_gate_number.add(had_counter_1[0])
                        remove_gate_number.add(had_counter_1[1])
                        had_counter_1 = []
                else:
                    had_counter_1 = []
        circuit = QuantumCircuit(n)
        for i, ele in enumerate(tmp_gates):
            if i not in remove_gate_number:
                circuit.add_gate(ele)
        return circuit


# def equivalence_full_check(circuit1, circuit2):
#     n = circuit1.get_qubit_count()
#     for i in range(2 ** (n)):
#         ket = QuantumState(n)
#         ket.set_computational_basis(i)
#         circuit1.update_quantum_state(ket)
#         bra = QuantumState(n)
#         bra.set_computational_basis(i)
#         circuit2.update_quantum_state(bra)
#         fidelity = abs(inner_product(bra, ket))
#         if fidelity < 0.95:
#             raise ValueError(f"Fidelity is incorrect: {fidelity}")
#             # print(f"Fidelity is incorrect: {fidelity}")
#     return fidelity
