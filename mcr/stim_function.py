import stim
from tqdm import tqdm

from mcr.rot_class import RotOps


def identity_to_under_score(pauli_string):
    values = pauli_string.replace("I", "_")
    return "".join(values)


# PauliStringをstimの言葉に変換する
def rot_ops_to_stim(rot_ops):
    original_pauli_string = rot_ops.get_pauli_strings()
    original_sgn = rot_ops.get_sgn_angles()
    stim_pauli_string = []
    for original_pauli_string, original_sgn in zip(original_pauli_string, original_sgn):
        tmp_string = identity_to_under_score(original_pauli_string)
        sign = "+" if original_sgn > 0 else "-"
        stim_pauli_string.append(stim.PauliString(f"{sign}{tmp_string}"))
    return stim_pauli_string


def stim_apply_pauli_gates(circuit, qubit_indices, pauli_ids, right_side=False):
    """指定された量子ビットに対してPauliゲートを適用"""
    # print(qubit_indices, pauli_ids)
    pauli_ids = pauli_ids[1:]  # 先頭は符号の情報なので除外
    pauli_ids = [pauli_ids[i] for i in qubit_indices]
    assert len(pauli_ids) == len(qubit_indices)
    for idx, pauli_id in zip(qubit_indices, pauli_ids):
        if pauli_id == "X":  # X
            circuit.append("H", [idx])
        elif pauli_id == "Y":  # Y
            if right_side:
                circuit.append("H", [idx])
                circuit.append("S", [idx])
            else:
                circuit.append("S_DAG", [idx])
                circuit.append("H", [idx])


def stim_conversion_from_pauli_to_circuit(stim_pauli):
    # 多分ここが違う
    sgn = stim_pauli.sign
    # Pauli演算子が入っているqubitのindexを取得
    non_identity_pauli_indices = stim_pauli.pauli_indices()
    position = max(non_identity_pauli_indices)  # 最後の非ゼロのindex
    circuit = stim.Circuit()

    stim_apply_pauli_gates(circuit, non_identity_pauli_indices, str(stim_pauli), right_side=False)
    if len(non_identity_pauli_indices) >= 2:  # CNOT必要
        # print("non_identity_pauli: ", non_identity_pauli_indices)

        # CNOTゲートを適用
        for idx in non_identity_pauli_indices:
            if idx != position:
                circuit.append("CNOT", [idx, position])

    # 回転ゲートを適用
    if sgn.real > 0:
        circuit.append("S", [position])
    else:
        circuit.append("S_DAG", [position])

    if len(non_identity_pauli_indices) >= 2:  # CNOT必要
        # CNOTゲートを逆順に適用
        for idx in reversed(non_identity_pauli_indices):
            if idx != position:
                circuit.append("CNOT", [idx, position])

    # Pauliゲートを逆順に適用
    stim_apply_pauli_gates(circuit, non_identity_pauli_indices, str(stim_pauli), right_side=True)
    return circuit


def get_rotation_relation(pauli_string_1, pauli_string_2):
    if str(pauli_string_1)[1:] == str(pauli_string_2)[1:]:
        if pauli_string_1.sign + pauli_string_2.sign == 0:
            return "zero"
        elif pauli_string_1.sign + pauli_string_2.sign == 2:
            return "plus_clifford"
        else:
            return "minus_clifford"
    elif pauli_string_1.commutes(pauli_string_2):
        return "commute"
    else:
        return "anti-commute"


# 持っているClifford circuitを元にPauli stringを更新する
def clifford_update(clifford_circuit, stim_pauli_str):
    # print(f"clifford updated!!: {stim_pauli_str}")
    # print(f"circuit: {clifford_circuit}")

    # inv = clifford_circuit.inverse()
    # tmp = stim_pauli_str.after(clifford_circuit)
    # return tmp.before(inv)
    new_val = stim_pauli_str.after(clifford_circuit)
    # print(f"after :{new_val}")
    return new_val


def optimization_process(target_str: stim.PauliString, data_all: list[stim.PauliString]):
    for i, ele in enumerate(reversed(data_all)):
        relation = get_rotation_relation(target_str, ele)
        if relation == "commute":
            continue
        elif relation == "anti-commute":
            # print("Anti-commute!!")
            return "Nothing"
        elif relation == "zero":
            # print("Delete_zero!", len(data_all) - i - 1, ele)
            # print(len(data_all))
            data_all = [data_all[j] for j in range(len(data_all)) if j != len(data_all) - i - 1]
            # print(len(data_all))
            return data_all
        elif relation in {"plus_clifford", "minus_clifford"}:
            # print("Delete_for_clifford!!", len(data_all) - i - 1, ele)
            # print(data_all)
            data_all = [data_all[j] for j in range(len(data_all)) if j != len(data_all) - i - 1]
            # print(data_all)
            # print(f"be clifford: {target_str}")
            new_clifford_circuit = stim_conversion_from_pauli_to_circuit(
                target_str
            )  # mergeされたPauliRotationを追加する機構が必要！
            return data_all, new_clifford_circuit
    return "Nothing"


# 最適化前のstim_rotsと最適化後のclifford+stim_rotsが一致するかqulacsのゲートに変換して検証する
def stim_to_rot_ops(stim_lst):
    new_rot_ops = []
    sgns = []
    for value in stim_lst:
        string = str(value)
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


def stim_rots_to_qulacs(stim_lst, nqubits):
    non_clifford_data = []
    for ele in stim_lst:
        if len(ele) < nqubits:
            diff = nqubits - len(ele)
            ele = stim.PauliString(ele + "_" * diff)
            non_clifford_data.append(ele)
        else:
            non_clifford_data.append(ele)
    rots = stim_to_rot_ops(non_clifford_data)
    return rots.convert_to_clifford_t_circuit(complement_id=False)


# T layer に分割するアルゴリズムの実装(arXiv:2407.08695 Algorithm 1)
def stim_grouping_of_pauli_rotations(stim_data_lst, joint=False):
    length = len(stim_data_lst)
    # print('length', length)
    L = []  # 空のリスト L
    for Rp in stim_data_lst:
        # print('Rp', Rp)
        j = 0  # 初期化（反可換なグループが見つからない場合は新しい層を作る）
        for k in reversed(range(len(L))):
            # print('k', k)
            # 反可換かどうかを判定
            commute_info = [Rp.commutes(Rk) for Rk in L[k]]
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
    if joint:
        return [item for sublist in L for item in sublist]  # Lの全要素をフラットにして返す
    else:
        return L


def zhang_optimization(stim_data_lst):
    clifford_circuit = stim.Circuit()
    optimized_rotations = []
    for ele in tqdm(reversed(stim_data_lst), desc="Zhang optimization", leave=False):
        # print(ele)
        if len(clifford_circuit) > 0:
            # print('clifford_circuit updated!!')
            ele = clifford_update(clifford_circuit, ele)
            # print('After: ', ele)
        if len(optimized_rotations) == 0:
            optimized_rotations.append(ele)

        else:
            # すでにappendされているPauliRotationとの関係を調べる
            # print(optimized_rotations[0])
            value = optimization_process(ele, optimized_rotations)
            if len(value) == 2 and isinstance(value, tuple):  # +pi/2 or -pi/2のClifford circuitが新たに生成された場合
                # print(value)
                # print('Clifford Merged!')
                # print('Bef',len(optimized_rotations))
                optimized_rotations, new_clifford_circuit = value
                # print('Aft',len(optimized_rotations))
                # print('New Cliff:',new_clifford_circuit)
                clifford_circuit = clifford_circuit + new_clifford_circuit
                # clifford_circuit = new_clifford_circuit + clifford_circuit

            else:  # len(value) == 1
                if value == "Nothing":
                    optimized_rotations.append(ele)

                else:  # 角度がちょうど打ち消しあって0になった場合
                    # print('Optimized!!!')
                    optimized_rotations = value
        # print(len(optimized_rotations))
        # print('-'*10)

    # 最後にreversedしておく
    optimized_rotations = optimized_rotations[::-1]
    # print(optimized_rotations, len(clifford_circuit))
    return optimized_rotations, clifford_circuit


def zhang_optimization_until_convergence(nqubits, stim_data_lst, with_grouping_t_layers=False, with_process=False):
    length = len(stim_data_lst)
    # print(length)
    clifford_data = []
    flag = True
    counter = 1
    non_clifford_gate_counts = []
    while flag:
        non_clifford_gate_counts.append(len(stim_data_lst))
        # print(f"{counter}th optimization, input length: {len(stim_data_lst)}")
        if with_grouping_t_layers:
            opt_rots_lst = []
            cliff_circs = []
            t_layers = stim_grouping_of_pauli_rotations(stim_data_lst, joint=False)
            # print(t_layers)
            # print(len(t_layers))
            for t_layer in t_layers:
                # print(t_layer)
                opt_rots, cliff = zhang_optimization(t_layer)
                opt_rots_lst.append(opt_rots)
                cliff_circs.append(cliff)
            # cliff_circs内の全ての回路を合成する & 回路のupdate
            combined_clifford_circuit = stim.Circuit()
            for i, circuit in enumerate(cliff_circs):
                # if len(circuit) > 0:
                #     print('update!')
                combined_clifford_circuit += circuit
                for j in range(i):
                    target_non_cliffords = opt_rots_lst[j]
                    opt_rots_lst[j] = [ele.after(circuit) for ele in target_non_cliffords]
            clifford_circuit = combined_clifford_circuit
            optimized_rotations = [
                item for sublist in opt_rots_lst for item in sublist
            ]  # ここでもう一度リストを平坦化する
        #####
        else:
            optimized_rotations, clifford_circuit = zhang_optimization(stim_data_lst)

        if len(optimized_rotations) >= length:
            flag = False
        if len(clifford_circuit) > 0:
            clifford_data.append(clifford_circuit)
        counter += 1
        stim_data_lst = optimized_rotations.copy()
        # stim_data_lst = stim_grouping_of_pauli_rotations(optimized_rotations, joint=True) # groupingをかける時
        # print('Grouping done!')
        length = len(optimized_rotations)
    # clifford_dataを左から順番に足す
    combined_clifford_circuit = stim.Circuit()
    for circuit in clifford_data:
        combined_clifford_circuit += circuit
    # combined_clifford_circuitに全ての回路が結合される
    if with_process:
        return optimized_rotations, combined_clifford_circuit, non_clifford_gate_counts
    else:
        return optimized_rotations, combined_clifford_circuit
