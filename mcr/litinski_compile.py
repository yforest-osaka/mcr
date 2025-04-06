import ast
import random
import re
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from qulacs import QuantumCircuit
from tqdm import tqdm

from mcr.pauli_bit_ops import (
    multiply_pauli_bits,
    pauli_bit_to_pauli_id,
    pauli_bit_to_pauli_string,
)
from mcr.rot_class import RotOps, satisfies_litinski_condition


def get_all_t_gate_positions(circuit):
    t_positions = []
    clifford_positions = []
    for i in range(circuit.get_gate_count()):
        if circuit.get_gate(i).get_name() == "T" or circuit.get_gate(i).get_name() == "Tdag":
            t_positions.append(i)
        else:
            clifford_positions.append(i)
    return t_positions, clifford_positions


# Litinski compile後のTゲートの回転ゲートを取得する
def t_swapper(circuit: QuantumCircuit, position: int):
    CLIFFORD_MAP_1Q = pd.read_csv("../data/1q.csv", index_col=0)
    CLIFFORD_MAP = pd.read_csv("../data/2q.csv", index_col=0)
    nqubits = circuit.get_qubit_count()
    target_gate = circuit.get_gate(position)
    target_gate_name = target_gate.get_name()
    if target_gate_name not in ["T", "Tdag"]:
        raise ValueError(f"T or Tdagである必要があります: {target_gate_name}")
    target_index = target_gate.get_target_index_list()[0]
    gate_count = circuit.get_gate_count()
    subsequent_gates = []
    for i in range(position + 1, gate_count):
        gate = circuit.get_gate(i)
        if gate.get_name() not in ["T", "Tdag"]:
            subsequent_gates.append(gate)

    # スタート地点を決める
    element = ["I"] * nqubits
    # z_position番目を'Z'に変更
    element[target_index] = "Z"

    if target_gate_name == "T":
        sgn = 1
    else:  # Tdag
        sgn = -1
    for clifford_gate in subsequent_gates:
        # print(element)
        clifford_gate_name = clifford_gate.get_name()
        if clifford_gate_name == "CNOT":
            ctrl = clifford_gate.get_control_index_list()[0]
            tg = clifford_gate.get_target_index_list()[0]
            if tg - ctrl < 0:
                clifford_gate_name = "CNOT_rev"  # 名称変更

        if clifford_gate_name in ["CNOT", "CNOT_rev", "CZ"]:
            ctrl = clifford_gate.get_control_index_list()[0]
            tg = clifford_gate.get_target_index_list()[0]
            if tg - ctrl < 0:
                ctrl, tg = tg, ctrl

            # print(f"Apply {clifford_gate_name}")
            target_2qubit_element = element[ctrl] + element[tg]
            new_element = CLIFFORD_MAP[clifford_gate_name][
                target_2qubit_element
            ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
            if "-" in new_element:
                # print("sgn change!")
                sgn *= -1
                new_element = new_element[1:]
            # 取得したnew_elementを元のelementに反映させる
            element[ctrl] = new_element[0]
            element[tg] = new_element[1]
        else:
            tg = clifford_gate.get_target_index_list()[0]  # 挟みたいclifford gateのindex
            # print(f'Apply {clifford_gate_name} on {tg}')
            target_1qubit_element = element[tg]
            # print(CLIFFORD_MAP_1Q[f"{clifford_gate_name}"])
            new_element = CLIFFORD_MAP_1Q[f"{clifford_gate_name}"][
                target_1qubit_element
            ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
            if "-" in new_element:
                # print("sgn change!")
                sgn *= -1
                new_element = new_element[1:]
            element[tg] = new_element[0]
    #     print('New elem: ', element)
    # print("after ", element)
    return "".join(element), sgn


def t_swapper_old(circuit: QuantumCircuit, position: int):
    CLIFFORD_MAP = pd.read_csv("../data/clifford_map.csv", index_col=0)
    nqubits = circuit.get_qubit_count()
    target_gate = circuit.get_gate(position)
    target_gate_name = target_gate.get_name()
    if target_gate_name not in ["T", "Tdag"]:
        raise ValueError(f"T or Tdagである必要があります: {target_gate_name}")
    target_index = target_gate.get_target_index_list()[0]
    gate_count = circuit.get_gate_count()
    subsequent_gates = []
    for i in range(position + 1, gate_count):
        gate = circuit.get_gate(i)
        if gate.get_name() not in ["T", "Tdag"]:
            subsequent_gates.append(gate)

    # スタート地点を決める
    tmp = ["I"] * nqubits
    # z_position番目を'Z'に変更
    tmp[target_index] = "Z"
    # リストを文字列に変換して返す
    element = "".join(tmp)

    if target_gate_name == "T":
        sgn = 1
    else:  # Tdag
        sgn = -1
    # print('before ', element)
    for clifford_gate in subsequent_gates:
        clifford_gate_name = clifford_gate.get_name()
        if clifford_gate_name == "CNOT":
            ctrl = clifford_gate.get_control_index_list()[0]
            tg = clifford_gate.get_target_index_list()[0]
            if tg - ctrl < 0:
                clifford_gate_name = "CNOT_rev"  # 名称変更

        if clifford_gate_name in ["CNOT", "CNOT_rev", "CZ"]:
            # print(f'Apply {clifford_gate_name}')
            new_element = CLIFFORD_MAP[clifford_gate_name][
                element
            ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
        else:
            tg = clifford_gate.get_target_index_list()[0]  # 挟みたいclifford gateのindex
            # print(f'Apply {clifford_gate_name}{tg}')
            new_element = CLIFFORD_MAP[f"{clifford_gate_name}{tg}"][
                element
            ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]

        if "-" in new_element:
            sgn *= -1
            element = new_element[1:]
        else:
            element = new_element
    #     print("New elem: ", element)
    # print("after ", element)
    return element, sgn


# T layer に分割するアルゴリズムの実装(arXiv:2407.08695 Algorithm 1)
def grouping_of_pauli_rotations(rot):
    length = len(rot.get_pauli_strings())
    # print('length', length)
    L = []  # 空のリスト L
    for Rp in range(length):
        # print('Rp', Rp)
        j = 0  # 初期化（反可換なグループが見つからない場合は新しい層を作る）
        for k in reversed(range(len(L))):
            # print('k', k)
            # 反可換かどうかを判定
            commute_info = [rot.is_commute(Rp, Rk) for Rk in L[k]]
            if not all(commute_info):  # 一つでも反可換があれば
                j = k + 1
                # print(f"all commuteじゃない: {j}")
                break
            else:
                # print(f'All commute! {k}')
                pass
        if j == 0:
            # 新しいグループを作成
            if len(L) == 0:
                L.append([Rp])
            else:
                L[0].append(Rp)
        else:
            # print(f"jが-1じゃない: {j}")
            if len(L) == j:
                # 新しい層を作成
                L.append([Rp])
            else:
                # 既存のグループに追加
                L[j].append(Rp)
    return L


# Litinski T layerの情報を元に、回転ゲートの順序を並び替えた新しいRotOpsを生成する
# 生成されるのは1個のRotOps(複数のLitinski T layerを含む)
def gen_replaced_rotation(rot, layer_info):
    pauli_ids = rot.get_pauli_ids()
    angles = rot.get_angles()
    new_pauli_ids = []
    new_angles = []
    for layer in layer_info:
        for number in layer:
            new_pauli_ids.append(pauli_ids[number])
            new_angles.append(angles[number])
    new_rot = RotOps(new_pauli_ids)
    new_rot.insert_angles(new_angles)
    return new_rot


# 1個のLitinski T layerの情報を元に、RotOpsを生成する(このレイヤー内の回転ゲートは互いにcommute)
def gen_t_layer(rot, one_layer_info):
    pauli_ids = rot.get_pauli_ids()
    angles = rot.get_angles()
    new_pauli_ids = []
    new_angles = []
    for number in one_layer_info:
        new_pauli_ids.append(pauli_ids[number])
        new_angles.append(angles[number])
    new_rot = RotOps(new_pauli_ids)
    new_rot.insert_angles(new_angles)
    return new_rot


def litinski_check(rot, layer_info, show_log=True):
    if len(layer_info) == 1:
        print("There exists only one layer!!")

    elements = []
    for layer in layer_info:
        new_rot = gen_t_layer(rot, layer)
        elements.append(new_rot)
    for num in range(len(elements) - 1):
        # print(num, num+1)
        rot1 = elements[num]
        rot2 = elements[num + 1]
        assert satisfies_litinski_condition(rot1, rot2), "Litinski condition is not satisfied"
        if show_log:
            print("OK")


def get_t_layer_from_clifford_t_circuit(circuit, show_log=False):
    # Litinski compile後のTゲートの回転ゲートを取得する
    t_positions, _ = get_all_t_gate_positions(circuit)
    if show_log:
        swapped_data = [
            t_swapper(circuit, position) for position in tqdm(t_positions, desc="swapping T gates", leave=False)
        ]
    else:
        swapped_data = [t_swapper(circuit, position) for position in t_positions]
    # print("swap_fin")
    rotation_strings = [ele[0] for ele in swapped_data]
    rotation_sgns = [ele[1] for ele in swapped_data]
    r = RotOps(rotation_strings)
    r.insert_angles_from_sgn(rotation_sgns)
    return r


def extract_clifford_circuit(circuit):
    # Cliffordゲートだけを取り出す
    _, clifford_positions = get_all_t_gate_positions(circuit)
    # print("swap_fin")
    clifford_circuit = QuantumCircuit(circuit.get_qubit_count())
    for position in clifford_positions:
        gate = circuit.get_gate(position)
        clifford_circuit.add_gate(gate)
    return clifford_circuit


def get_rots_after_pauli_grouping(rot_ops):
    t_layer_info = grouping_of_pauli_rotations(rot_ops)

    # if len(rot_ops) < 100:
    #     litinski_check(rot_ops, t_layer_info, show_log=False)

    # if len(t_layer_info) == 1:
    #     for _ in tqdm(range(1000000)):
    #         i, j = random.sample(range(len(t_layer_info[0])), 2)
    #         assert rot_ops.is_commute(i, j)
    results = [gen_t_layer(rot_ops, t_layer) for t_layer in t_layer_info if len(t_layer) > 0]
    return results


# localにoptimizeする関数
# 互いにcommuteするような1個のT layerを回転軸のデータだけを元に最適化する
# (Cliffordには非対応)
def sum_angles_for_strings(pauli_strings, angles, new_method=False):
    angle_dict = defaultdict(float)  # defaultdictを使うことで角度を0度に初期化できる
    for string, angle in zip(pauli_strings, angles):
        angle_dict[string] += angle
    # new_rot = RotOps(dict(angle_dict).keys())
    # 計算結果回転角が0でないものだけを取り出す
    result = {key: value for key, value in angle_dict.items() if value != 0}
    if len(result) == 0:
        return None
    pauli_ids = list(result.keys())
    angles = list(result.values())

    non_clifford_paulis, clifford_paulis, paulis = [], [], []
    non_clifford_angles, clifford_angles, pauli_angles = [], [], []
    for pauli, angle in zip(pauli_ids, angles):
        modulus = int((angle / (np.pi / 4)) % 8)
        # assert modulus != 0
        if modulus == 0:
            continue
        ang = np.pi / 4
        if modulus == 1:
            non_clifford_paulis.append(pauli)
            non_clifford_angles.append(ang)
        elif modulus == 2:
            clifford_paulis.append(pauli)
            clifford_angles.append(2 * ang)
        elif modulus == 3:
            non_clifford_paulis.append(pauli)
            clifford_paulis.append(pauli)
            non_clifford_angles.append(ang)
            clifford_angles.append(2 * ang)
        elif modulus == 4:
            clifford_paulis.append(pauli)
            clifford_angles.append(4 * ang)
        elif modulus == 5:
            non_clifford_paulis.append(pauli)
            clifford_paulis.append(pauli)
            non_clifford_angles.append(-1 * ang)
            clifford_angles.append(-2 * ang)
        elif modulus == 6:
            clifford_paulis.append(pauli)
            clifford_angles.append(-2 * ang)
        elif modulus == 7:
            non_clifford_paulis.append(pauli)
            non_clifford_angles.append(-1 * ang)
        else:
            raise ValueError(f"modulusが不正です: {modulus}")
    if len(non_clifford_paulis) > 0:
        non_clifford_rot = RotOps(non_clifford_paulis)
        non_clifford_rot.insert_angles(non_clifford_angles)
    else:
        non_clifford_rot = []

    if len(clifford_paulis) > 0:
        clifford_rot = RotOps(clifford_paulis)
        clifford_rot.insert_angles(clifford_angles)
    else:
        clifford_rot = []

    # if len(paulis) > 0:
    #     pauli_rot = RotOps(paulis)
    #     pauli_rot.insert_angles(pauli_angles)
    # else:
    #     pauli_rot = []

    if not new_method:
        rot = RotOps([])
        if len(non_clifford_rot) > 0:
            rot.merge(non_clifford_rot)
        if len(clifford_rot) > 0:
            rot.merge(clifford_rot)
        # if len(pauli_rot) > 0:
        #     rot.merge(pauli_rot)
        return rot
    else:
        # return non_clifford_rot, clifford_rot, pauli_rot
        return non_clifford_rot, clifford_rot


# 共通の回転軸を持つ回転ゲートをまとめて最適化する
def optimize_t_layer(rot_lst, new_method):
    results = []
    for ele in rot_lst:
        d = sum_angles_for_strings(ele.get_pauli_strings(), ele.get_angles(), new_method)
        test_rot = RotOps([])
        if d is not None:
            if new_method:
                for r in d:
                    if len(r) > 0:
                        test_rot.merge(r)
                # from unoptimization.circuit_ops import (
                #     are_unitaries_equivalent,
                #     get_merged_matrix,
                # )

                # print("check")
                # m = get_merged_matrix(test_rot.convert_to_clifford_t_circuit())
                # m2 = get_merged_matrix(ele.convert_to_clifford_t_circuit())
                # assert are_unitaries_equivalent(
                #     m, m2
                # ), "Unitaries are not equivalent when separating clifford non-clifford layer!"
            results.append(d)
        # else:
        #     print("None!")
    return results


def all_merge_rot(rot_lst):
    all_rot = RotOps(rot_lst[0].get_pauli_strings())
    all_rot.insert_angles(rot_lst[0].get_angles())
    for rot in rot_lst[1:]:
        all_rot.merge(rot)
    return all_rot


def is_pauli_bit_commute(pauli_bit1: int, pauli_bit2: int) -> bool:
    tuple_bits1, tuple_bits2 = pauli_bit1[1:], pauli_bit2[1:]
    sgn = 1
    for i in range(len(tuple_bits1)):
        target = [tuple_bits1[i], tuple_bits2[i]]
        if target[0] != target[1] and (0, 0) not in target:  # anti-commute
            sgn *= -1
    if sgn == 1:
        return True
    return False


def multiply_coefficient(pauli_bit, coefficient):
    if coefficient == 1.0j:
        result = [int((pauli_bit[0] + 1) % 4)] + list(pauli_bit[1:])
        return tuple(result)
    elif coefficient == -1.0j:
        result = [int((pauli_bit[0] + 3) % 4)] + list(pauli_bit[1:])
        return tuple(result)
    elif coefficient == -1:
        result = [int((pauli_bit[0] + 2) % 4)] + list(pauli_bit[1:])
        return tuple(result)
    else:
        raise ValueError(f"The coefficient is not supported.: {coefficient}")


def update_non_clifford(non_clifford_ops, clifford_ops):
    # print('input')
    # print(non_clifford_ops.get_pauli_strings())
    # print(non_clifford_ops.get_angles())
    # print(clifford_ops.get_pauli_strings())
    # print(clifford_ops.get_angles())
    # print(non_clifford_ops.get_pauli_strings() == sample_info)
    # print(non_clifford_ops.get_angles() == sample_info_angles)
    # print(clifford_ops.get_pauli_strings() == sample_cl_strs)
    # print(clifford_ops.get_angles() == sample_cl_angles)
    # print('length objects: ',len(clifford_ops))
    initial_rot = non_clifford_ops.duplicate()
    initial_rot.merge(clifford_ops)

    if len(non_clifford_ops) == 0:
        # print("non clifford is empty")
        return non_clifford_ops
    tmp = non_clifford_ops.get_pauli_bits()
    for i in range(len(clifford_ops)):
        new_bits = []
        target_clifford = clifford_ops.extract(i, i + 1)
        clifford_bit = target_clifford.get_pauli_bits()[0]
        # print("Target clifford", pauli_bit_to_pauli_string(clifford_bit))
        # non-clifford layerの更新
        for index, non_clifford_bit in enumerate(tmp):
            # print(non_clifford_bit)
            if not is_pauli_bit_commute(non_clifford_bit, clifford_bit):
                modulus = int((target_clifford.get_angles()[0] / (np.pi / 4)) % 8)
                if modulus in [2, 6]:
                    # print('index: ',index)
                    # print("before: ", pauli_bit_to_pauli_string(non_clifford_bit))
                    new_val = multiply_pauli_bits(non_clifford_bit, clifford_bit)  # P'Pへ変換
                    if modulus == 2:
                        # print("after: ", pauli_bit_to_pauli_string(multiply_coefficient(new_val, 1j)))
                        new_bits.append(multiply_coefficient(new_val, 1j))  # i倍してappend
                    else:  # modulus == 6
                        # print("after: ", pauli_bit_to_pauli_string(multiply_coefficient(new_val, -1j)))
                        # print("check:", multiply_coefficient(new_val, -1j)[0])
                        new_bits.append(multiply_coefficient(new_val, -1j))  # -i倍してappend
                    # print("-" * 10)
                else:
                    assert modulus == 4, f"The modulus is not 4.: {modulus}"
                    new_bits.append(multiply_coefficient(non_clifford_bit, -1))  # -1倍してappend
                    # print("-" * 10)
            else:
                new_bits.append(non_clifford_bit)
        assert len(new_bits) == len(tmp), "The length of the new bits is not the same as the original one."
        tmp = new_bits
    # この時点で new_bitsの回転軸にiや-iが含まれている場合はエラーを出力する。-1の場合はそれに対応するangleを-1倍する
    change_angle_info = [ele[0] for ele in new_bits]
    # print("change_angle_info: ", change_angle_info)
    # print([pauli_bit_to_pauli_id(ele) for ele in new_bits])

    original_angles = non_clifford_ops.get_angles()
    # print(non_clifford_ops.get_sgn_angles())
    output_bits = []
    new_angles = []
    # i = 0
    for num in range(len(change_angle_info)):
        number = change_angle_info[num]
        angle = original_angles[num]
        if number == 0:
            new_angles.append(angle)
            output_bits.append(new_bits[num])
        elif number == 2:
            # print('Found!: ',i)
            new_angles.append(-1 * angle)  # 角度を-1倍
            output_bits.append(multiply_coefficient(new_bits[num], -1))  # その代わりにpauli_bitsを-1倍

        else:
            if number == 1:
                raise ValueError(f"Rotation axis has an imaginary unit!: 1j")
            elif number == 3:
                raise ValueError(f"Rotation axis has an imaginary unit!: -1j")
            else:
                raise ValueError(f"Invalid number!: {number}")
        i += 1
    result_ops = RotOps([pauli_bit_to_pauli_id(ele) for ele in output_bits])
    result_ops.insert_angles(new_angles)
    # print("converted: non-clifford: ", result_ops.get_all())
    # print(non_clifford_ops.get_sgn_angles() == result_ops.get_sgn_angles())
    # from unoptimization.circuit_ops import are_unitaries_equivalent, get_merged_matrix

    # m = get_merged_matrix(initial_rot.convert_to_clifford_t_circuit())
    # tmp_rot = clifford_ops.duplicate()
    # tmp_rot.merge(result_ops)
    # m2 = get_merged_matrix(tmp_rot.convert_to_clifford_t_circuit())
    # assert are_unitaries_equivalent(m, m2), "Unitaries are not equivalent when swapping clifford non-clifford layer!"

    return result_ops  # 回転軸・角度を更新したnon_clifford回路だけを返す


def optimize_until_convergence(rot, with_clifford_update=True, show_log=False, new_method=False):
    data_x, data_tcount = [], []
    i = 0
    data_x.append(i)
    data_tcount.append(len(rot))
    i += 1
    if with_clifford_update:
        if show_log:
            print(f"{i} th optimization")
        swapped_clifford = RotOps([])
        while True:
            if show_log:
                print("grouping")
                data = get_rots_after_pauli_grouping(rot)
                print("grouping finished")
            else:
                data = get_rots_after_pauli_grouping(rot)
            non_clifford_layer = RotOps([])
            clifford_layer = RotOps([])

            optimized_data = optimize_t_layer(
                data, new_method=True
            )  # この時点で完全にゲートがなくなるT layerも存在するのでlengthは小さくなる

            for l in tqdm(range(len(optimized_data)), desc="Optimizing T layers", leave=False):
                # print("L", l)
                current_non_clifford, current_clifford = optimized_data[l]
                start = len(non_clifford_layer) + len(clifford_layer)
                addition = len(current_non_clifford) + len(current_clifford)
                # print(f"add_info: non-Clifford: {len(current_non_clifford)}, Clifford: {len(current_clifford)}")
                # print("existing_non_clifford", non_clifford_layer.get_all())
                if not isinstance(current_clifford, list):  # non-cliffordを更新
                    # print("should update clifford!")
                    if len(non_clifford_layer) > 0:
                        for_test = non_clifford_layer.duplicate()
                        # print("update clifford")
                        # print("current_clifford: ", current_clifford.get_all())
                        # print("existing_non_clifford_before_update", non_clifford_layer.get_all())
                        non_clifford_layer = update_non_clifford(non_clifford_layer, current_clifford)
                        assert len(for_test) == len(
                            non_clifford_layer
                        ), "The length of the non-clifford layer is not the same."
                    clifford_layer.merge(current_clifford)
                    # print(f"Clifford added!: {len(clifford_layer)}")

                if len(current_non_clifford) > 0:  # non-cliffordが空の場合も存在する
                    non_clifford_layer.merge(current_non_clifford)
                    # print("non-Clifford added!")
                end = len(non_clifford_layer) + len(clifford_layer)
                # print("end: ", end)

                assert (
                    end == start + addition
                ), f"Length!: {l+1}th iteration, start:{start}, addition:{addition}, end:{end}"
                # print("final_existing_non_clifford", non_clifford_layer.get_all())
                # print("---" * 10)

            if len(clifford_layer) > 0:
                swapped_clifford.merge(clifford_layer)
                # print(f"swapped_clifford added!: {len(swapped_clifford)}")

            if (
                non_clifford_layer.get_non_clifford_gate_count() >= rot.get_non_clifford_gate_count()
            ):  # 最適化後のnon-clifford部分のみのRot数が最適化前の全体Rot数以上になる場合は終了
                if show_log:
                    print("finish")
                break
            data_x.append(i)
            data_tcount.append(non_clifford_layer.get_non_clifford_gate_count())
            rot = non_clifford_layer.duplicate()
            i += 1
        swapped_clifford.merge(non_clifford_layer)  # 最後にnon-clifford_layerを右にマージする
        return swapped_clifford, data_x, data_tcount

    else:
        while True:
            data = get_rots_after_pauli_grouping(rot)  # T layerが何個か生成される
            new_rot_lst = optimize_t_layer(data, new_method)  # 各layerごとに最適化を行う
            new_rot = all_merge_rot(new_rot_lst)  # new_rot_lstの情報をマージする機構を追加する
            if new_rot.get_non_clifford_gate_count() == rot.get_non_clifford_gate_count():
                if show_log:
                    print("finish")
                break
            data_x.append(i)
            data_tcount.append(new_rot.get_non_clifford_gate_count())
            rot = new_rot.duplicate()
            i += 1
        return new_rot, data_x, data_tcount


def initialize_t_gate(data, t_position, qubit_count):
    gate_name, qubits = data[t_position]
    gate_data = []
    if gate_name.lower() == "t":
        for i in range(qubit_count):
            if i == qubits[0]:
                gate_data.append((1, 0))
            else:
                gate_data.append((0, 0))
        return (0, *gate_data)
    else:
        assert gate_name.lower() == "tdg", f"Gate name is not T or Tdg: {gate_name}"
        for i in range(qubit_count):
            if i == qubits[0]:
                gate_data.append((1, 0))
            else:
                gate_data.append((0, 0))
        return (2, *gate_data)


def get_non_identity_qubit_indices(pauli_bit):
    pauli_bit = pauli_bit[1:]
    return [i for i, elem in enumerate(pauli_bit) if elem != (0, 0)]


CLIFFORD_MAP_1Q = pd.read_csv("../data/bit_1q.csv", index_col=0)
CLIFFORD_MAP_2Q = pd.read_csv("../data/bit_2q.csv", index_col=0)


def t_swap(t_gate_pauli_bit, t_position, data_all, direction="right"):
    if direction not in ["right", "left"]:
        raise ValueError(f"Invalid direction: {direction}")
    if direction == "right":
        search_data = data_all[t_position + 1 :]
    else:
        search_data = data_all[:t_position][::-1]
    for elem in search_data:
        gate_name, qubit_lst = elem
        # print(gate_name, qubit_lst)
        if gate_name.lower() in ["t", "tdg"]:
            continue  # T or Tdag gateはスキップ
        else:
            current_sgn = t_gate_pauli_bit[0]
            current_pauli_bit = t_gate_pauli_bit[1:]

            current_non_idenity_qubit_indices = get_non_identity_qubit_indices(t_gate_pauli_bit)
            intersection_indices = set(current_non_idenity_qubit_indices) & set(qubit_lst)
            if len(intersection_indices) == 0:  # 共通するqubitが1つもない場合→何もしなくて良い
                # print('なし')
                continue
            elif gate_name in ["x", "y", "z", "h", "s", "sdg"]:  # 共通するqubitが1つ場合→1qubit Pauliの更新が必要

                tg = qubit_lst[0]  # 挟みたいclifford gateのindex

                # S, Sdagゲートで右から左へスワップしたい時はmapが少し変更される
                if gate_name == "s" and direction == "left":
                    gate_name = "sdg"
                if gate_name == "sdg" and direction == "left":
                    gate_name = "s"

                bit_to_pauli_string = {(0, 0): "I", (0, 1): "X", (1, 1): "Y", (1, 0): "Z"}
                new_element = CLIFFORD_MAP_1Q[f"{gate_name}"][
                    bit_to_pauli_string[current_pauli_bit[tg]]
                ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
                new_element = ast.literal_eval(new_element)
                if -1 in new_element:
                    current_sgn = (current_sgn + 2) % 4
                    new_element = new_element[1:]  # 符号の情報を除去
                new_pauli_bits = []
                for i, pauli_bit in enumerate(current_pauli_bit):
                    if i == tg:
                        new_pauli_bits.append(new_element[0])
                    else:
                        new_pauli_bits.append(pauli_bit)
            else:  # 共通するqubitが2つ以上ある場合→2qubit Pauliの更新が必要
                assert gate_name in ["cx", "cz", "swap"], f"Invalid gate name: {gate_name}"
                # print('qubit_lst:', qubit_lst)
                ctrl, tg = qubit_lst
                if tg - ctrl < 0 and gate_name == "cx":
                    gate_name = "cx_rev"
                    ctrl, tg = tg, ctrl
                # print("TEST:", (current_pauli_bit[ctrl], current_pauli_bit[tg]))
                bit_to_pauli_string = {
                    ((0, 0), (0, 0)): "II",
                    ((0, 0), (0, 1)): "IX",
                    ((0, 0), (1, 1)): "IY",
                    ((0, 0), (1, 0)): "IZ",
                    ((0, 1), (0, 0)): "XI",
                    ((0, 1), (0, 1)): "XX",
                    ((0, 1), (1, 1)): "XY",
                    ((0, 1), (1, 0)): "XZ",
                    ((1, 1), (0, 0)): "YI",
                    ((1, 1), (0, 1)): "YX",
                    ((1, 1), (1, 1)): "YY",
                    ((1, 1), (1, 0)): "YZ",
                    ((1, 0), (0, 0)): "ZI",
                    ((1, 0), (0, 1)): "ZX",
                    ((1, 0), (1, 1)): "ZY",
                    ((1, 0), (1, 0)): "ZZ",
                }

                new_element = CLIFFORD_MAP_2Q[f"{gate_name}"][
                    bit_to_pauli_string[(current_pauli_bit[ctrl], current_pauli_bit[tg])]
                ]  # 交換した後に出てくるelementを出力する関数を記述 # CLIFFORD_MAP[両端に入るゲート][真ん中のゲート]
                new_element = ast.literal_eval(new_element)
                # print(type(new_element))
                if -1 in new_element:
                    current_sgn = (current_sgn + 2) % 4
                    new_element = new_element[1:]  # 符号の情報を除去
                new_pauli_bits = []
                for i, pauli_bit in enumerate(current_pauli_bit):
                    if i == ctrl:
                        new_pauli_bits.append(new_element[0])
                    elif i == tg:
                        new_pauli_bits.append(new_element[1])
                    else:
                        new_pauli_bits.append(pauli_bit)
            t_gate_pauli_bit = (current_sgn, *new_pauli_bits)
            # print(t_gate_pauli_bit)
            print(get_non_identity_qubit_indices(t_gate_pauli_bit))
    return pauli_bit_to_pauli_string(t_gate_pauli_bit)


def litinski_compile_circuit(circuit_file, direction="right", cpu_count=1):
    st = time()
    if isinstance(circuit_file, QuantumCircuit):  # qulacsの回路が入力されたと想定
        from qulacs.converter import convert_qulacs_circuit_to_QASM

        string = convert_qulacs_circuit_to_QASM(circuit_file)
    else:  # qasmファイルが入力されたと想定
        with open(circuit_file, mode="r") as f:
            string = f.read().splitlines()
    circ_qasm = [line for line in string if line != ""]

    for ele in circ_qasm:
        if "qreg" in ele:
            qubit_count = int(re.findall(r"\d+", ele)[0])
            break

    exclude = ["OPENQASM", "include", "qreg", "creg", "measure"]
    circ_qasm = [line for line in circ_qasm if not any([word in line for word in exclude])]

    # Define a regular expression pattern to match gate names and qubit numbers
    pattern = re.compile(r"(\w+)\s+q\[(\d+)\](?:,q\[(\d+)\])?;")

    # Extract gate names and qubit numbers
    extracted_gates = []
    for line in circ_qasm:
        match = pattern.match(line)
        if match:
            gate_name = match.group(1)
            qubits = [int(match.group(2))]
            if match.group(3):
                qubits.append(int(match.group(3)))
            extracted_gates.append((gate_name, qubits))

    t_positions = []
    for i, (gate_name, qubits) in enumerate(extracted_gates):
        if gate_name.lower() == "t" or gate_name.lower() == "tdg":
            t_positions.append(i)
    print("gate_count", len(extracted_gates))
    print("t_count", len(t_positions))
    print("time:", time() - st)
    # results = []
    results = Parallel(n_jobs=cpu_count)(
        delayed(t_swap)(
            initialize_t_gate(extracted_gates, t_position, qubit_count),
            t_position,
            extracted_gates,
            direction=direction,
        )
        for t_position in tqdm(t_positions, desc="Litinski compile...", leave=False)
    )
    return results
