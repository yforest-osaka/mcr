import copy
import os
import random
import re
from itertools import product
from uuid import uuid4

import numpy as np
import pandas as pd
import pyzx as zx
from mqt import qcec
from mqt.qcec.configuration import augment_config_from_kwargs
from mqt.qcec.pyqcec import Configuration
from qulacs import QuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
from qulacs.gate import (  # type: ignore
    CNOT,
    RZ,
    DenseMatrix,
    H,
    Identity,
    ParametricPauliRotation,
    ParametricRZ,
    RandomUnitary,
    S,
    Sdag,
    T,
    Tdag,
    X,
    Y,
    Z,
)
from qulacs_core import (
    ClsOneControlOneTargetGate,
    ClsOneQubitGate,
    QuantumGate_SingleParameter,
)
from tqdm import tqdm

from mcr.filesave import qulacs_to_qasm
from mcr.mycircuit import GateFactory, MyQuantumProgram, get_gate_info


def get_merged_matrix(circuit: QuantumCircuit | MyQuantumProgram) -> np.ndarray:
    """Qulacの回路を受け取って、全体のユニタリ行列を返す

    Args:
        circuit (QuantumCircuit): qulacsの回路

    Returns:
        np.ndarray: 全体のユニタリ行列
    """
    if isinstance(circuit, MyQuantumProgram):
        qulacs_circ = circuit.set_circuit()
    else:
        qulacs_circ = circuit.copy()
    QCO().optimize(qulacs_circ, qulacs_circ.get_qubit_count())
    assert qulacs_circ.get_gate_count() == 1, f"マージに失敗しました: {qulacs_circ.get_gate_count()}"
    return qulacs_circ.get_gate(0).get_matrix()


def parametric_pauli_to_dem(circuit):
    new_circuit = QuantumCircuit(circuit.get_qubit_count())
    for i in range(circuit.get_gate_count()):
        gate = circuit.get_gate(i)
        if "Parametric" in gate.get_name():
            target = gate.get_target_index_list()
            matrix = gate.get_matrix()
            new_circuit.add_gate(DenseMatrix(target, matrix))
        else:
            new_circuit.add_gate(gate)
    return new_circuit


def get_lowerest_2q_gate(QuantumProgram: MyQuantumProgram) -> list[int]:
    """QuantumProgramの中から一番下のqubit_indexに所属するゲートのindexを取り出す

    Args:
        QuantumProgram (MyQuantumProgram): 量子回路

    Returns:
        list[int]: 一番下のqubit_indexに所属するゲートのindexが入ったリスト
    """
    n = QuantumProgram.get_qubit_count()
    lowerest_index = n - 1
    candidates = []
    for i in range(QuantumProgram.get_gate_count()):
        tg = QuantumProgram[i][1].get_target_index_list()
        if lowerest_index in tg:
            candidates.append(i)
    return candidates


def set_to_my_quantum_program(qulacs_circuit: QuantumCircuit, reverse_option=False) -> MyQuantumProgram:
    """qulacsの回路を受け取ってMyQuantumProgramを返す

    Args:
        qulacs_circuit (QuantumCircuit): qulacsの回路
        reverse_option (bool, optional): ゲートの配置順を反転させるかどうか. Defaults to False.

    Returns:
        MyQuantumProgram: 量子回路
    """
    n = qulacs_circuit.get_qubit_count()
    circ = MyQuantumProgram(n)
    for i in range(qulacs_circuit.get_gate_count()):
        if reverse_option:
            circ.add_gate((qulacs_circuit.get_gate_count() - i - 1,), qulacs_circuit.get_gate(i))
        else:
            circ.add_gate((i,), qulacs_circuit.get_gate(i))
    return circ


def get_all_elements_from_label(data: list, label_number: tuple[int]) -> list[int]:
    """回路データから該当するラベルのものだけを抽出する

    Args:
        data (list): 量子回路のデータリスト
        label_number (tuple[int]): 抽出したいラベルの番号

    Returns:
        list[int]: 該当するラベルのpositionが入ったリスト
    """
    results = []
    for i, ele in enumerate(data):
        if ele[1] == label_number:
            results.append(i)
    return results


def gen_gate_from_gatename(
    gate_name_str: str, index: int, angle: float | bool = False
) -> ClsOneQubitGate | QuantumGate_SingleParameter:
    """ゲートの名前とindexから1qubitのqulacs gateを生成する

    Args:
        gate_name_str (str): ゲートの名前
        index (int): ターゲットのindex
        angle (float | bool, optional): 角度. Defaults to False.

    Raises:
        ValueError: ParametricRZの場合はangleを指定してください

    Returns:
        ClsOneQubitGate | QuantumGate_SingleParameter: 1qubitのqulacs gate
    """
    if not angle:
        gate_dicts = {
            "I": Identity(index),
            "X": X(index),
            "Y": Y(index),
            "Z": Z(index),
            "H": H(index),
            "S": S(index),
            "Sdag": Sdag(index),
            "T": T(index),
            "Tdag": Tdag(index),
        }
        return gate_dicts[gate_name_str]
    else:
        assert "Parametric" in gate_name_str
        if not angle:
            raise ValueError("ParametrixRZ: angleを指定してください")
        return ParametricRZ(index, angle)


def get_maximum_index(gate_list: list) -> int:
    """gate_listの中に入っているゲートの中で最大のqubit番号を取得する

    Args:
        gate_list (list): ゲートのリスト

    Returns:
        int: 最大のqubit番号
    """
    result = []
    for gate in gate_list:
        result += gate.get_control_index_list() + gate.get_target_index_list()
    return max(result)


def map_seed_circuit(
    circuit: list | QuantumCircuit, new_indices: list[int], max_qubit_count: int = 2
) -> QuantumCircuit:
    """種となるqulacs回路と行き先のindexを指定して回路を書き換える。最大の量子ビット数も必要。

    Args:
        circuit (list | QuantumCircuit): qulacsの回路
        new_indices (list[int]): 新しい行き先のリスト
        max_qubit_count (int, optional): 最大の量子ビット数. Defaults to 2.

    Returns:
        QuantumCircuit: 書き換えられた回路
    """
    # assert len(new_indices) == 2
    # circuitがgate_listで与えられた場合の処理
    if type(circuit) == list:
        n = get_maximum_index(circuit) + 1
    else:
        n = circuit.get_qubit_count()
    circuit_out = QuantumCircuit(max_qubit_count)
    dictionary = dict(zip([i for i in range(n)], new_indices))

    # 新しい回路へゲートを配置する
    if type(circuit) == list:
        circuit_out = []
        for gate in circuit:
            target_indices = gate.get_target_index_list()
            target = [dictionary[tg] for tg in target_indices]
            name = gate.get_name()
            if name != "I":
                if name == "CNOT":
                    target = dictionary[gate.get_target_index_list()[0]]
                    control = dictionary[gate.get_control_index_list()[0]]
                    circuit_out.append(CNOT(control, target))
                elif name == "DenseMatrix":
                    circuit_out.append(DenseMatrix(target, gate.get_matrix()))
                elif name == "ParametricRZ":  # ParametricRZを追加
                    phase = gate.get_parameter_value()
                    circuit_out.append(ParametricRZ(target[0], phase))
                else:
                    circuit_out.append(gen_gate_from_gatename(name, target[0]))
    else:  # QuantumCircuitの場合
        for i in range(circuit.get_gate_count()):
            gate = circuit.get_gate(i)
            target_indices = gate.get_target_index_list()
            target = [dictionary[tg] for tg in target_indices]
            name = gate.get_name()
            if name != "I":
                if name == "CNOT":
                    target = dictionary[gate.get_target_index_list()[0]]
                    control = dictionary[gate.get_control_index_list()[0]]
                    circuit_out.add_gate(CNOT(control, target))
                elif name == "DenseMatrix":
                    circuit_out.add_gate(DenseMatrix(target, gate.get_matrix()))
                else:
                    circuit_out.add_gate(gen_gate_from_gatename(name, target[0]))
    return circuit_out


def remove_identity(circuit: QuantumCircuit) -> QuantumCircuit:
    """qulacsの回路中に含まれるIdentityを除去する

    Args:
        circuit (QuantumCircuit): qulacsの回路

    Returns:
        QuantumCircuit: Identityを除去した回路
    """
    n = circuit.get_qubit_count()
    circuit_out = QuantumCircuit(n)
    for i in range(circuit.get_gate_count()):
        gate = circuit.get_gate(i)
        name = gate.get_name()
        if name != "I":
            target = gate.get_target_index_list()[0]
            if name == "CNOT":
                control = gate.get_control_index_list()[0]
                circuit_out.add_CNOT_gate(control, target)
            else:
                circuit_out.add_gate(gen_gate_from_gatename(name, target))
    return circuit_out


# ゲート数、CNOT, Tの割合を指定して2qubitのrandom clifford+t回路を作る
# def generate_2q_clifford_t_circuit(gate_count, p_cnot=0.2, p_t=0.2)
# QulacsGateSequenceGeneratorのgenerate_clifford_circuitに合併


def is_commutable(
    gate_left: ClsOneQubitGate | ClsOneControlOneTargetGate, gate_right: ClsOneQubitGate | ClsOneControlOneTargetGate
) -> bool:
    """2つのqulacs gateが可換かを判定する

    Args:
        gate_left (ClsOneQubitGate | ClsOneControlOneTargetGate): 量子ゲート
        gate_right (ClsOneQubitGate | ClsOneControlOneTargetGate): 量子ゲート

    Returns:
        bool: 可換かどうか
    """
    left_gate_name = gate_left.get_name()
    left_gate_ctrl_index = gate_left.get_control_index_list()
    left_gate_tg_index = gate_left.get_target_index_list()
    right_gate_name = gate_right.get_name()
    right_gate_ctrl_index = gate_right.get_control_index_list()
    right_gate_tg_index = gate_right.get_target_index_list()

    target_indices_all = len(
        set(left_gate_ctrl_index + left_gate_tg_index + right_gate_ctrl_index + right_gate_tg_index)
    )
    flag = False
    # CNOTが含まれるかどうかで4パターンに場合分け
    if left_gate_name == "CNOT":
        if right_gate_name == "CNOT":  # 両方ともCNOT
            if target_indices_all >= 4:
                flag = True
            else:
                if left_gate_ctrl_index != right_gate_ctrl_index and left_gate_tg_index != right_gate_tg_index:
                    flag = False
                else:
                    flag = True
        else:  # 左側だけCNOT
            if target_indices_all >= 3:
                flag = True
            else:  # 1qubit gateのtgとCNOTのctrlが一致しており、右側のゲートがアダマール以外の場合は可換
                if left_gate_ctrl_index == right_gate_tg_index and right_gate_name != "H":
                    flag = True
    else:
        if right_gate_name == "CNOT":  # 右側だけCNOT
            if target_indices_all >= 3:
                flag = True
            else:  # 1qubit gateのtgとCNOTのctrlが一致しており、左側のゲートがアダマール以外の場合は可換
                if left_gate_tg_index == right_gate_ctrl_index and left_gate_name != "H":
                    flag = True
        else:  # どちらもSingle qubit gate
            if target_indices_all >= 2:
                flag = True
            else:  # 左右のtarget_indexが一致しており(当然)、両方のゲートがアダマール以外(S, Sdag, T, Tdag)の場合は可換
                assert left_gate_tg_index == right_gate_tg_index
                if left_gate_name != "H" and right_gate_name != "H":
                    flag = True
                elif left_gate_name == "H" and right_gate_name == "H":  # ただし両方ともアダマールの場合は可換
                    flag = True
    return flag


def dagger(gate_name_str: str) -> str:
    """ゲートの名前からダガーに相当するゲート名を取得する

    Args:
        gate_name_str (str): ゲートの名前

    Returns:
        str: ダガーに相当するゲート名
    """
    dictionary = {
        "I": "I",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        "H": "H",
        "S": "Sdag",
        "Sdag": "S",
        "T": "Tdag",
        "Tdag": "T",
        "ParametricRZ": "ParametricRZ",
    }
    return dictionary[gate_name_str]


def get_inverse_circuit_gate_list(tmp_gates: list) -> list:
    """qulacsのゲートリストを逆順にしてリストで返す

    Args:
        tmp_gates (list): qulacsのゲートリスト

    Raises:
        ValueError: [description]

    Returns:
        list: 逆順になったゲートリスト
    """
    inv_tmp_gates = []
    for gate in reversed(tmp_gates):
        try:
            gate_name = gate.get_name()
            tg = gate.get_target_index_list()[0]
            if gate_name != "ParametricRZ":
                inv_tmp_gates.append(gen_gate_from_gatename(dagger(gate_name), tg))
            else:
                phase = gate.get_parameter_value()
                inv_tmp_gates.append(gen_gate_from_gatename(gate_name, tg, -1 * phase))
        except:
            gate_name = gate.get_name()
            if gate_name == "CNOT":
                tg = gate.get_target_index_list()[0]
                ctrl = gate.get_control_index_list()[0]
                inv_tmp_gates.append(CNOT(ctrl, tg))
            else:
                raise ValueError(f"{gate_name}")
    return inv_tmp_gates


def single_qubit_gate_swap(gate: ClsOneQubitGate, another_gate: ClsOneQubitGate) -> list:
    """左側の1qubit gateが右側と可換でない場合に計算し直し 生成されるゲート列を返す

    Args:
        gate (ClsOneQubitGate): 1qubit ゲート
        another_gate (ClsOneQubitGate): 1qubit ゲート

    Raises:
        ValueError: CNOTゲートはswapできません

    Returns:
        list: 生成されたゲート列
    """
    # gateの形はそのまま another_gateをgenerated_gatesとして出力する
    gate_name = gate.get_name()
    # print('Gate name: ', gate_name, gate.get_target_index_list())
    if gate_name == "CNOT":
        raise ValueError("CNOTゲートはswapできません")
    another_gate_name = another_gate.get_name()

    # もしゲートのどちらかがParametricRZだった場合はangle情報を取得しておく
    gate_angle, another_gate_angle = None, None
    if gate_name == "ParametricRZ":
        gate_angle = gate.get_parameter_value()
    if another_gate_name == "ParametricRZ":
        another_gate_angle = another_gate.get_parameter_value()

    if another_gate_name != "CNOT":  # A^dagger B Aを返す(量子回路上に起こす時には積の順番が逆になることに注意)
        assert (
            gate.get_target_index_list()[0] == another_gate.get_target_index_list()[0]
        )  # qubit_indexが同じかをチェック
        tg = gate.get_target_index_list()[0]
        generated_gates = []

        generated_gates.append(gen_gate_from_gatename(gate_name, tg, gate_angle))
        generated_gates.append(gen_gate_from_gatename(another_gate_name, tg, another_gate_angle))
        if gate_name == "ParametricRZ":
            generated_gates.append(gen_gate_from_gatename(dagger(gate_name), tg, angle=-1 * gate_angle))
        else:
            generated_gates.append(gen_gate_from_gatename(dagger(gate_name), tg))

    else:  # gateがanother_gateのtarget_index側に入っている場合(もしくはgateがアダマールで、another_gateのcontrol_indexに入っている場合)
        tg = gate.get_target_index_list()[0]
        cnot_ctrl = another_gate.get_control_index_list()[0]
        cnot_tg = another_gate.get_target_index_list()[0]
        generated_gates = []
        generated_gates.append(gen_gate_from_gatename(gate_name, tg, angle=gate_angle))
        generated_gates.append(CNOT(cnot_ctrl, cnot_tg))

        if gate_name == "ParametricRZ":
            generated_gates.append(gen_gate_from_gatename(dagger(gate_name), tg, angle=-1 * gate_angle))
        else:
            generated_gates.append(gen_gate_from_gatename(dagger(gate_name), tg))
    return generated_gates


def get_inversed_param_rz(gate: QuantumGate_SingleParameter) -> QuantumGate_SingleParameter:
    """回転ゲートの回転角を反転させたゲートを返す

    Args:
        gate (QuantumGate_SingleParameter): ParametricRZゲート

    Returns:
        QuantumGate_SingleParameter: 回転角を反転させたParametricRZゲート
    """
    assert gate.get_name() == "ParametricRZ"
    tg = gate.get_target_index_list()[0]
    phase = gate.get_parameter_value()
    return ParametricRZ(tg, -1 * phase)


def cnot_gate_swap(
    cnot_gate: ClsOneControlOneTargetGate, another_gate: ClsOneQubitGate | ClsOneControlOneTargetGate
) -> list:
    """CNOT gateが右側と可換でない場合に計算し直し 生成されるゲート列を返す

    Args:
        cnot_gate (ClsOneControlOneTargetGate): CNOTゲート
        another_gate (ClsOneQubitGate | ClsOneControlOneTargetGate): 1qubit ゲート or CNOTゲート

    Raises:
        ValueError: 1qubitゲートはswapできません

    Returns:
        list: 生成されたゲート列
    """
    # cnot_gateの形はそのまま another_gateをgenerated_gatesとして出力する
    generated_gates = []
    gate_name = cnot_gate.get_name()
    cnot_tg = cnot_gate.get_target_index_list()[0]
    cnot_ctrl = cnot_gate.get_control_index_list()[0]
    if gate_name != "CNOT":
        raise ValueError("1qubitゲートはswapできません")
    another_gate_name = another_gate.get_name()
    if another_gate_name != "CNOT":  # 1qubitゲートの場合
        single_tg = another_gate.get_target_index_list()[0]
        if another_gate.get_name() != "ParametricRZ":
            generated_gates = [
                CNOT(cnot_ctrl, cnot_tg),
                gen_gate_from_gatename(another_gate.get_name(), single_tg),
                CNOT(cnot_ctrl, cnot_tg),
            ]
        else:
            assert another_gate.get_name() == "ParametricRZ"
            generated_gates = [
                CNOT(cnot_ctrl, cnot_tg),
                ParametricRZ(single_tg, another_gate.get_parameter_value()),
                CNOT(cnot_ctrl, cnot_tg),
            ]
    else:  # CNOTゲート同士の入れ替え
        another_cnot_tg = another_gate.get_target_index_list()[0]
        another_cnot_ctrl = another_gate.get_control_index_list()[0]
        if cnot_tg == another_cnot_ctrl and cnot_ctrl == another_cnot_tg:  # swapされた状態(特別なケース)
            generated_gates = [
                CNOT(another_cnot_tg, another_cnot_ctrl),
                CNOT(another_cnot_ctrl, another_cnot_tg),
                CNOT(another_cnot_tg, another_cnot_ctrl),
            ]
        elif cnot_tg == another_cnot_ctrl:  # cnot_gateのtargetとanother_gateのcontrolのindexが同じ
            assert cnot_ctrl != another_cnot_tg
            generated_gates = [
                CNOT(cnot_ctrl, cnot_tg),
                CNOT(another_cnot_ctrl, another_cnot_tg),
                CNOT(cnot_ctrl, cnot_tg),
            ]
        else:  # cnot_gateのcontrolとanother_gateのtargetのindexが同じ
            assert cnot_ctrl == another_cnot_tg
            assert cnot_tg != another_cnot_ctrl
            generated_gates = [
                CNOT(cnot_ctrl, cnot_tg),
                CNOT(another_cnot_ctrl, another_cnot_tg),
                CNOT(cnot_ctrl, cnot_tg),
            ]
    return generated_gates


def list_to_quantum_program(data: list[tuple], qubit_count: int) -> MyQuantumProgram:
    """リスト[(id,gate)]で書かれた変数をMyQuantumProgramに変換する

    Args:
        data (list[tuple]): MyQuantumProgramに変換したいリスト(gate_idとgateのtuple)
        qubit_count (int): 量子ビット数

    Returns:
        MyQuantumProgram: 量子回路
    """
    c = MyQuantumProgram(qubit_count)
    for i, ele in enumerate(data):
        try:
            c.add_gate(ele[0], ele[1])
        except:
            c.add_gate((i,), ele)
    return c


def qulacs_to_my_quantum_program(qulacs_circuit: QuantumCircuit, ids: list[tuple]) -> MyQuantumProgram:
    """qulacsの回路とidのタプルが入ったリストを引数にしてMyQuantumProgramを作る

    Args:
        qulacs_circuit (QuantumCircuit): qulacsの回路
        ids (list[tuple]): ゲートIDのリスト

    Raises:
        ValueError: ゲート数とゲートIDの個数が一致しない場合

    Returns:
        MyQuantumProgram: 量子回路
    """
    n = qulacs_circuit.get_qubit_count()
    n_gates = qulacs_circuit.get_gate_count()
    if len(ids) != n_gates:
        raise ValueError(f"ゲート数とゲートIDの個数が一致しません ids: {len(ids)}, gate_count: {n_gates}")
    circuit = MyQuantumProgram(n)
    for i, id in enumerate(ids):
        circuit.add_gate(id, qulacs_circuit.get_gate(i))
    return circuit


def swap_two_circuits(left_qc: QuantumCircuit, right_qc: QuantumCircuit, swap_prob: float) -> tuple[list, list, list]:
    """2つの回路をswapする。swapする割合を指定することもできる。戻り値は3つの回路

    Args:
        left_qc (QuantumCircuit): 量子回路
        right_qc (QuantumCircuit): 量子回路
        swap_prob (float): swapする割合

    Raises:
        ValueError: swap_probを適切な値に設定してください

    Returns:
        tuple[list, list, list]: swapによって作られた3つの回路(left, center, right)
    """
    if not (0 < swap_prob <= 1):
        raise ValueError(f"swap_probを適切な値に設定してください: {swap_prob}")

    if type(left_qc) == list:
        gate_group_a = left_qc
    else:
        gate_group_a = [left_qc.get_gate(g) for g in range(left_qc.get_gate_count())]
    if type(right_qc) == list:
        gate_group_b = right_qc
    else:
        gate_group_b = [right_qc.get_gate(g) for g in range(right_qc.get_gate_count())]

    swap_gates_count = int(len(gate_group_a) * (1 - swap_prob))
    gate_position_for_swap = list(reversed(range(swap_gates_count, len(gate_group_a))))

    results = gate_group_b.copy()

    for idx in tqdm(gate_position_for_swap, desc="swapping_circuits"):
        base_gate = gate_group_a[idx]
        new_results = []

        for gate_b in results:
            if is_commutable(base_gate, gate_b):
                new_results.append(gate_b)
            else:
                if base_gate.get_name() == "CNOT":
                    new_results.extend(cnot_gate_swap(base_gate, gate_b))
                else:
                    new_results.extend(single_qubit_gate_swap(base_gate, gate_b))
        results = new_results

    unchanged_left_circuit = gate_group_a[:swap_gates_count]
    changed_left_circuit = gate_group_a[swap_gates_count:]

    return unchanged_left_circuit, results, changed_left_circuit


def are_unitaries_equivalent(
    U: np.ndarray, V: np.ndarray, up_to_global_phase: bool = True, inverse_relation: bool = False
) -> bool:
    """2つのユニタリ行列が等価(or逆行列関係)かどうかを判定する

    Args:
        U (np.ndarray): 行列
        V (np.ndarray): 行列
        up_to_global_phase (bool, optional): グローバル位相の違いを許可するか. Defaults to True.
        inverse_relation (bool, optional): 逆行列の関係にあるかどうかを調べたいときはTrueにする. Defaults to False.

    Returns:
        bool: 判定結果
    """
    if inverse_relation:  # 逆行列の関係にあるかどうかを調べたいとき
        W = np.dot(U, V)
    else:  # 等価であるかを調べたいとき
        # U のエルミート共役（共役転置）を計算
        U_dagger = np.conj(U.T)
        W = np.dot(U_dagger, V)

    # W が対角行列かどうかを確認
    is_diagonal = np.allclose(W, np.diag(np.diag(W)))

    if not is_diagonal:
        return False

    # 対角成分が同じ位相因子かどうかを確認
    diagonal_elements = np.diag(W)
    phases = np.angle(diagonal_elements)
    phases = np.where(phases == -np.pi, np.pi, phases)  # -np.piの位相が出てきた時はnp.piに置き換える(同一視してOK)
    is_same_phase = np.allclose(phases, phases[0])

    if not up_to_global_phase:  # グローバル位相のズレも許容しない場合のオプション
        return np.allclose(phases, 0)

    return is_same_phase


def delete_parametric_rotation_from_circuit(MyQC: MyQuantumProgram) -> MyQuantumProgram:
    """ParametricRZで定義していた回路(MyQuantumProgram)をRotZに変換する関数を作る(そうしないとqasmファイルに変換できない)

    Args:
        MyQC (MyQuantumProgram): ParametricRZで定義していた回路

    Returns:
        MyQuantumProgram: ParametricRZをRZに変換した回路
    """
    n = MyQC.get_qubit_count()
    circuit = MyQuantumProgram(n)
    circuit.index_distribution = copy.deepcopy(MyQC.index_distribution)  # index_distributionのcopy
    circuit.del_nums = MyQC.del_nums.copy()  # del_numsのコピー

    for ele in tqdm(MyQC):
        if ele[1].get_name() == "ParametricRZ":
            phase = ele[1].get_parameter_value()
            gate_id = ele[0]
            tg = ele[1].get_target_index_list()[0]
            circuit.add_gate(gate_id, RZ(tg, phase))
        else:
            circuit.add_gate(ele[0], ele[1])
    return circuit


# CLIFFORD_MAP_1Q = pd.DataFrame()
# # 'X','Y','Z'の順で行き先を登録。CLIFFORD_MAP_1Qの右側に書いてある文字で挟んだ時の行き先を考える
# CLIFFORD_MAP_1Q["X"] = ["X", "-Y", "-Z"]
# CLIFFORD_MAP_1Q["Y"] = ["-X", "Y", "-Z"]
# CLIFFORD_MAP_1Q["Z"] = ["-X", "-Y", "Z"]
# CLIFFORD_MAP_1Q["H"] = ["Z", "-Y", "X"]
# CLIFFORD_MAP_1Q["S"] = ["Y", "-X", "Z"]
# CLIFFORD_MAP_1Q["Sdag"] = ["-Y", "X", "Z"]
# CLIFFORD_MAP_1Q = CLIFFORD_MAP_1Q.rename(index={0: "X", 1: "Y", 2: "Z"})


def show_clifford_map_1q() -> pd.DataFrame:
    CLIFFORD_MAP_1Q = pd.read_csv("../data/clifford_map_1q.csv", index_col=0)
    return CLIFFORD_MAP_1Q


def conjugation_1q(gate_name: str, index: int) -> ClsOneQubitGate:
    """ゲート名から1qubitのCliffordゲートを返す

    Args:
        gate_name (str): ゲート名
        index (int): ターゲットのindex

    Returns:
        ClsOneQubitGate: 1qubitのCliffordゲート
    """
    gate_dicts = {
        "X": X(index),
        "Y": Y(index),
        "Z": Z(index),
        "H": H(index),
        "S": Sdag(index),
        "Sdag": S(index),
    }
    return gate_dicts[gate_name]


def t_swapper_2q(circuit: MyQuantumProgram, position: int):  # 2qubit回路限定!!(今後は使わない)
    CLIFFORD_MAP = pd.read_csv("../data/clifford_map.csv", index_col=0)
    target_gate = circuit[position][1]
    target_gate_name = target_gate.get_name()
    if target_gate_name not in ["T", "Tdag"]:
        raise ValueError(f"T or Tdagである必要があります: {target_gate_name}")
    target_index = target_gate.get_target_index_list()[0]
    subsequent_gates = [
        ele for ele in circuit[position + 1 :] if ele[1].get_name() not in ["T", "Tdag"]
    ]  # 計算変換が必要になるゲートたち
    if target_index == 0:
        # element = ['Z', 'I']
        element = "ZI"
    else:
        # element = ['I', 'Z']
        element = "IZ"
    if target_gate_name == "T":
        # sgn = [1, 1]
        sgn = 1
    else:  # Tdag
        # sgn = []
        # for ele in element:
        #     if ele == 'I':
        #         sgn.append(1)
        #     else:
        #         sgn.append(-1)
        sgn = -1
    # print('before: ',element, sgn)
    for clifford_gate in subsequent_gates:
        clifford_gate_name = clifford_gate[1].get_name()
        if clifford_gate_name == "CNOT":
            ctrl = clifford_gate[1].get_control_index_list()[0]
            tg = clifford_gate[1].get_target_index_list()[0]
            if tg - ctrl < 0:
                clifford_gate_name = "CNOT_rev"  # 名称変更

        if clifford_gate_name in ["CNOT", "CNOT_rev", "CZ"]:
            # print(f'Apply {clifford_gate_name}')
            new_element = CLIFFORD_MAP[clifford_gate_name][
                element
            ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
        else:
            tg = clifford_gate[1].get_target_index_list()[0]  # 挟みたいclifford gateのindex
            # print(f'Apply {clifford_gate_name}{tg}')
            new_element = CLIFFORD_MAP[f"{clifford_gate_name}{tg}"][
                element
            ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]

        if "-" in new_element:
            sgn *= -1
            element = new_element[1:]
        else:
            element = new_element
    return element, sgn


# Unoptimizationに必要な関数(2024/10/21追加)
# position=0が一番使われていないqubit, position=-1が一番使われているqubit
def get_most_used_qubit_index(circuit, except_qubit_indices=[], position: int = 0):
    qubit_dist = circuit.index_distribution.copy()
    # print(qubit_dist)
    if except_qubit_indices:
        for i in except_qubit_indices:
            qubit_dist.pop(str(i))
    sorted_keys = sorted(qubit_dist, key=lambda k: qubit_dist[k])
    # print(sorted_keys)
    # print("selected:", int(sorted_keys[position]))
    return int(sorted_keys[position])


def choose_one_target_qubit_of_left_gate(circuit, arg_index_select_option):
    nqubits = circuit.get_qubit_count()
    if arg_index_select_option == "equally":
        min_count_index = get_most_used_qubit_index(circuit, position=0)  # 一番使われていないqubitを選ぶ
    else:  # random choice
        min_count_index = np.random.randint(0, nqubits)
    return min_count_index


def search_gates_from_qubit_indices(qubit_indices, circuit, first_only=False):
    gate_numbers = []
    for i, gate_data in enumerate(list(circuit)):
        if i not in circuit.del_nums:
            target_index = gate_data[1].get_target_index_list()
            tf_lst = [qubit_index in target_index for qubit_index in qubit_indices]
            if any(tf_lst):
                gate_numbers.append(i)
                if first_only:
                    return gate_numbers
    return gate_numbers


def assign_gate_id_of_generated_gates(gate_lst, base_id):
    gates = []
    # 新しく生成されたゲートに割り当てるgate idを作る
    base_gate_id = list(base_id)
    for i, gate in enumerate(gate_lst):  # \tilde{Udagger}とAを入れる
        generated_gate_id = tuple(base_gate_id + [i])
        gates.append((generated_gate_id, gate))
    return gates


def check_arg_circuit_type(arg_circuit_type):
    if arg_circuit_type not in ["RandomUnitary", "Clifford_t"]:
        raise ValueError(f"the argument arg_circuit_type is incorrect: {arg_circuit_type}")
    return arg_circuit_type


def check_arg_index_select_option(arg_index_select_option):
    if arg_index_select_option not in ["equally", "random"]:
        raise ValueError(f"the argument arg_index_select_option is incorrect: {arg_index_select_option}")
    return arg_index_select_option


def check_arg_swap_option(arg_swap_option):
    if arg_swap_option not in ["keep_left", "keep_right", "random"]:
        raise ValueError(f"the argument arg_swap_option is incorrect: {arg_swap_option}")
    return arg_swap_option


def check_arg_allow_3q_gate(arg_allow_3q_gate):
    if arg_allow_3q_gate not in [True, False]:
        raise ValueError(f"the argument arg_allow_3q_gate is incorrect: {arg_allow_3q_gate}")
    return arg_allow_3q_gate


def check_arg_run_kak(arg_run_kak):
    if arg_run_kak not in [True, False]:
        raise ValueError(f"the argument arg_run_kak is incorrect: {arg_run_kak}")
    return arg_run_kak


def gen_nontrivial_clifford_insertion_gates(
    qubit_indices, arg_gate_count, check_clifford=False
):  # 非自明なClifford列生成
    gate_u_list, gate_u_dagger_list = [], []
    assert arg_gate_count == 4

    candidates = [
        [[[0, 1], [1, 0]], [[2, 2], [3, 3]]],
        [[[0, 1], [2, 0]], [[1, 3], [3, 2]]],
        [[[0, 1], [3, 0]], [[1, 2], [2, 3]]],
        [[[0, 2], [1, 0]], [[2, 3], [3, 1]]],
        [[[0, 2], [2, 0]], [[1, 1], [3, 3]]],
        [[[0, 2], [3, 0]], [[1, 3], [2, 1]]],
        [[[0, 3], [1, 0]], [[2, 1], [3, 2]]],
        [[[0, 3], [2, 0]], [[1, 2], [3, 1]]],
        [[[0, 3], [3, 0]], [[1, 1], [2, 2]]],
        [[[1, 1], [2, 2]], [[1, 2], [2, 1]]],
        [[[1, 1], [2, 3]], [[1, 3], [2, 1]]],
        [[[1, 1], [3, 2]], [[1, 2], [3, 1]]],
        [[[1, 1], [3, 3]], [[1, 3], [3, 1]]],
        [[[1, 2], [2, 3]], [[1, 3], [2, 2]]],
        [[[1, 2], [3, 3]], [[1, 3], [3, 2]]],
        [[[2, 1], [3, 2]], [[2, 2], [3, 1]]],
        [[[2, 1], [3, 3]], [[2, 3], [3, 1]]],
        [[[2, 2], [3, 3]], [[2, 3], [3, 2]]],
    ]

    sequence = random.choice(candidates)
    left_sequence = random.sample(sequence, 2)
    sgn_count = random.choice([0, 2, 4])  # 偶数個minusを割り当てる
    left_sgns = random.sample([1] * sgn_count + [-1] * (4 - sgn_count), 4)  # 左側4個の符号
    all_sgns = left_sgns.copy()
    all_sgns.extend([-1 * s for s in left_sgns])  # 5番目,6番目の符号は左側の2個の符号を反転させたものにする

    # all_sgns.extend([-1 * s for s in left_sgns[:2]])  # 5番目,6番目の符号は左側の2個の符号を反転させたものにする
    # all_sgns.append(random.choice([1, -1]))  # 7番目の符号は1か-1にする(どちらでもいい)
    # all_sgns.append(random.choice([1, -1]))  # 8番目の符号は1か-1にする(どちらでもいい)
    all_sequence = []
    for ele in left_sequence:
        all_sequence.extend(list(ele))
    all_sequence *= 2

    all_gate_list = [
        ParametricPauliRotation(qubit_indices, pauli_ids, -np.pi / 4 * angle)
        for pauli_ids, angle in zip(all_sequence, all_sgns)
    ]
    gate_u_list = all_gate_list[:4]
    gate_u_dagger_list = all_gate_list[4:]

    # gate_u_list = [
    #     ParametricPauliRotation(qubit_indices, pauli_ids, -np.pi / 4 * angle)
    #     for pauli_ids, angle in zip(all_sequence, all_sgns)
    # ]
    # gate_u_dagger_list = [
    #     ParametricPauliRotation(qubit_indices, pauli_ids, np.pi / 4 * angle)
    #     for pauli_ids, angle in zip(all_sequence, all_sgns)
    # ]

    # gate_u_dagger_list = list(reversed(gate_u_dagger_list))
    if check_clifford:
        from mcr.clifford import is_clifford
        from mcr.filesave import qulacs_to_pyzx
        from mcr.mycircuit import MyQuantumProgram
        from mcr.pyzxfunc import optimize

        clifford_circ = MyQuantumProgram(max(qubit_indices) + 1)
        for i, gate in enumerate(gate_u_list + gate_u_dagger_list):
            clifford_circ.add_gate((i,), gate)
        rots = parametric_myqc_to_rot_ops(clifford_circ)
        tmp = rots.convert_to_clifford_t_circuit()
        c_zx = qulacs_to_pyzx(tmp)
        c_opt = optimize(c_zx)
        assert c_opt.tcount() > 0, "Locally optimized!!"
        assert is_clifford(c_opt.to_matrix()), "Not Clifford!!"
    return gate_u_list, gate_u_dagger_list


def gen_insertion_gates(circuit_type, qubit_indices, gate_count=1):
    if circuit_type == "RandomUnitary":  #! RandomUnitary Circuit
        gate_u = RandomUnitary(qubit_indices)  # この子のゲートidはまだ決まらない
        gate_u_dagger = DenseMatrix(qubit_indices, np.conjugate(gate_u.get_matrix()).T)  # この子のゲートidは付かない
        return gate_u, gate_u_dagger

    elif circuit_type == "Clifford_t":  #! Clifford + T circuit
        # ゲートのPauli回転軸を決定
        gate_u_list, gate_u_dagger_list = [], []
        for _ in range(gate_count):
            while True:
                # 重複を許して2つランダムに選ぶ
                pauli_ids = random.choices([0, 1, 2, 3], k=2)
                # 結果が[0, 0]であればやり直す
                if pauli_ids != [0, 0]:
                    break
            # arg_angles = [-np.pi / 2, -np.pi / 4, np.pi / 4, np.pi / 2]
            arg_angles = [-np.pi / 4, np.pi / 4]
            angle = random.choice(arg_angles)
            gate_u = ParametricPauliRotation(qubit_indices, pauli_ids, angle)  # この子のゲートidはまだ決まらない
            gate_u_dagger = ParametricPauliRotation(qubit_indices, pauli_ids, -1 * angle)
            gate_u_list.append(gate_u)
            gate_u_dagger_list.append(gate_u_dagger)
        gate_u_dagger_list = list(reversed(gate_u_dagger_list))
        gate_u = gate_u_list
        gate_u_dagger = gate_u_dagger_list
        return gate_u, gate_u_dagger


def unoptimization(
    arg_original_circuit,
    arg_unopt_iteration_count,
    arg_circuit_type,
    arg_index_select_option,
    arg_swap_option,
    arg_allow_3q_gate,
    arg_run_kak,
    arg_gate_count=1,  # U U daggerとして挿入するときのゲート数を決めることができる
    arg_insert_nontrivial_clifford=False,
):
    # check arguments
    arg_circuit_type = check_arg_circuit_type(arg_circuit_type)
    arg_index_select_option = check_arg_index_select_option(arg_index_select_option)
    arg_swap_option = check_arg_swap_option(arg_swap_option)
    arg_allow_3q_gate = check_arg_allow_3q_gate(arg_allow_3q_gate)
    arg_run_kak = check_arg_run_kak(arg_run_kak)

    circuit = arg_original_circuit.duplicate()
    nqubits = circuit.get_qubit_count()
    # maximum_number = 1000000
    for k in tqdm(range(arg_unopt_iteration_count), leave=False):
        # for k in range(arg_unopt_iteration_count):
        # print(f'iteration: {k}')
        # 非最適化する際に使用する qubit_indexをゲートの分布状況から1個決定する
        target_qubit_index = choose_one_target_qubit_of_left_gate(circuit, arg_index_select_option)

        # target_qubit_indexを含んでいるようなゲートの候補が全て出てくる
        candidate_gate_positions = search_gates_from_qubit_indices([target_qubit_index], circuit, first_only=False)

        left_gate_position = random.choice(candidate_gate_positions)  # 左側のゲートの位置を決定

        left_gate_id, left_gate_val = circuit[left_gate_position]
        # print('gate_id: ',left_gate_id)
        tg = left_gate_val.get_target_index_list()  # 左側のゲートのqubit位置

        gate1 = GateFactory(left_gate_val)  # このゲート(左側)を非最適化する
        # u_gateを挿入するもう１つのqubit_indexを決める(tgのindexを除いて一番使われていないものを選ぶ)
        index_for_insertion = get_most_used_qubit_index(circuit, except_qubit_indices=tg, position=0)

        assert index_for_insertion != target_qubit_index
        index = sorted([target_qubit_index, index_for_insertion])  # ここにu_gateを挿入する
        # print('target_qubit_index: ',target_qubit_index)
        # print('index_for_insertion: ',index_for_insertion)
        # print('index: ',index)

        # 挿入するゲートの種類を決定(将来的に非自明なClifford列生成ができる関数を作成)
        if arg_insert_nontrivial_clifford:
            # print("insert nontrivial clifford")
            gate_u, gate_u_dagger = gen_nontrivial_clifford_insertion_gates(index, arg_gate_count, check_clifford=False)
        else:
            gate_u, gate_u_dagger = gen_insertion_gates(arg_circuit_type, index, gate_count=arg_gate_count)

        circuit.delete(left_gate_position)  # 左側のゲートは新しいものに変化するのでここで削除

        # swapする時にどちらのゲートを更新するかについてのオプション
        if arg_swap_option == "random":
            opt = random.choice(["keep_left", "keep_right"])
        else:
            opt = arg_swap_option

        if arg_gate_count == 1:  # 従来通り
            single_gate_u = gate_u
            single_gate_u_dagger = gate_u_dagger

            """スワップの機構"""
            gate2 = GateFactory(single_gate_u_dagger)

            if arg_allow_3q_gate:
                gates = gate1.swap(
                    gate2, option=opt, run_kak=arg_run_kak, qubit_count=nqubits
                )  # For RandomUnitary(Use 3q decomposition)
            else:
                gates = gate1.swap_without_3q_gate(gate2, option=opt)  # For Clifford_T(Not Use 3q decomposition)
            gate_u = [gate_u]
        else:
            assert (
                arg_allow_3q_gate == False
            ), "複数個回転ゲートを挿入する非最適化は allow_3q_gate=True に設定できません"
            assert opt == "keep_left", "複数個回転ゲートを挿入する非最適化は keep_left の場合しか使用できません"

            gates = []
            if arg_insert_nontrivial_clifford:
                # 入れ替えを実行しない場合
                # print("test")
                gates.append(gate1.gate)
                gates += gate_u_dagger
            else:
                # 入れ替えを実行する場合
                gates = []
                for u_dag in gate_u_dagger:
                    gate2 = GateFactory(u_dag)
                    gates += gate1.swap_without_3q_gate(gate2, option=opt, remove_edge_gate=True)
                gates.append(gate1.gate)
                assert len(gates) == 3 * arg_gate_count + 1, "生成ゲート数が一致しません"

        ########
        # todo: 受け取ったgatesにidを付けてcircuitにappendする
        # 新しく生成されたゲートに関して gate idをつける

        new_generated_gates = assign_gate_id_of_generated_gates(gates + gate_u, left_gate_id)  # 変更(1129)
        if arg_circuit_type == "Clifford_t":
            assert arg_allow_3q_gate == False, "Clifford_t回路では allow_3q_gate=True は設定できません"
            # assert len(new_generated_gates) == 5
        for gate_id, gate in new_generated_gates:
            circuit.add_gate(gate_id, gate)

        # 次に非最適化するゲートを設定(強制的にconcatenateさせたい時などに使う)
        # next_target_gate_position = len(circuit)-2

    return arg_original_circuit, circuit


def parametric_myqc_to_rot_ops(myqc_circuit):
    from mcr.rot_class import RotOps

    pauli_info, angle_info = [], []
    nqubits = myqc_circuit.get_qubit_count()
    for ele in myqc_circuit.sort_gate_sequence():
        indices, pauli_strings, angle = get_gate_info(ele[1])

        paulis = ["I"] * nqubits
        for i, p in zip(indices, pauli_strings):
            paulis[i] = p
        pauli_info.append("".join(paulis))
        angle_info.append(-1 * angle)  # Parametric Pauli Rotationの角度は反転させないと等価にならないので注意
    rot = RotOps(pauli_info)
    rot.insert_angles(angle_info)
    return rot


def get_qubit_count_from_qasm_file(filepath):

    # ファイルを読み込む
    with open(filepath, "r") as file:
        content = file.read()

    # 正規表現でqregの横の数字を取得
    match = re.search(r"qreg q\[(\d+)\];", content)
    if match:
        number = match.group(1)
        return int(number)
    else:
        raise ValueError("qubit数が分かりませんでした")


def equivalence_check_via_mqt_qcec(circuit_1, circuit_2, exclude_zx_checker=False, show_log=True):
    # circuit_1, circuit_2はQASMファイルでもOK
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
            configuration, {"run_simulation_checker": True, "run_zx_checker": False, "timeout": 5}
        )
        result = qcec.verify(circuit_1, circuit_2, configuration=configuration).equivalence
    else:
        result = qcec.verify(circuit_1, circuit_2).equivalence
    if remove_flag1:
        os.remove(circuit_1)
    if remove_flag2:
        os.remove(circuit_2)
    if show_log:
        print(result.name)
    return result.name in {"equivalent", "equivalent_up_to_global_phase"}


def equivalence_via_pyzx(qasmfile_1, qasmfile_2):
    c_zx1 = zx.Circuit.from_qasm_file(qasmfile_1)
    c_zx2 = zx.Circuit.from_qasm_file(qasmfile_2)
    return c_zx1.verify_equality(c_zx2)
