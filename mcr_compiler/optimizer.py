# three_layer_nontrivial_swapを実行することはゲートを追加挿入してmcr_swapを実行することと同じ？
from mcr.gate_apply import PauliBit, grouping, loop_optimization, multiply_all
from mcr.mcr_optimize import find_mcr, find_nontrivial_swap
from mcr.equiv_check import (
    equiv,
    equivalence_check_via_mqt_qcec,
)
from mcr.rotation_circuit import PauliRotationSequence
from qulacs import QuantumCircuit
from mcr.gate_apply import set_clifford_to_qulacs

from time import time
import numpy as np
import pickle
from copy import deepcopy


def have_common_pauli_str(left_bits, right_bits):
    left_strs = {bit.get_pauli_str() for bit in left_bits}
    right_strs = {bit.get_pauli_str() for bit in right_bits}
    return not left_strs.isdisjoint(right_strs)


def contains_list(lst):
    return any(isinstance(item, list) for item in lst)


def mcr_swap(pauli_bit_groups, with_mcr_index=False, show_log=False):
    data = deepcopy(pauli_bit_groups)
    results = set()
    remove_index_set = set()
    for i in range(len(data) - 1):
        if i not in remove_index_set:
            left_data = data[i]
            right_data = data[i + 1]
            swappable_check = find_mcr(left_data, right_data)
            if swappable_check:
                if show_log:
                    print(f"Swapping {left_data} and {right_data} at index {i}")
                    # Uncomment for debugging
                results.add(i)
                data[i] = swappable_check[0]
                data[i + 1] = swappable_check[1]
                # if len(grouping(data[i + 1])) >= 1:  # any case?
                #     remove_index_set.add(i + 1)
                if (
                    len(data[i + 1]) >= 3 and len(grouping(data[i + 1])) >= 2
                ):  # 入れ替え後の右側の要素数が3個以上かつTレイヤー化した時のレイヤー数が2以上の時は次ループのswap対象から除外
                    remove_index_set.add(i + 1)
                elif i < len(data) - 2 and have_common_pauli_str(
                    data[i + 1], data[i + 2]
                ):
                    # print("have common pauli_str, removing next-next index")
                    remove_index_set.add(i + 2)
    new_data = sum(data, [])
    if with_mcr_index:
        return new_data, results
    return new_data


def check_basis_changeable(pauli_bit_groups):
    remove_index_set = set()
    results = set()
    for i in range(1, len(pauli_bit_groups) - 3):
        if i not in remove_index_set:
            left_data = pauli_bit_groups[i]
            center_data = pauli_bit_groups[i + 1]
            right_data = pauli_bit_groups[i + 2]
            swappable_check = find_nontrivial_swap(
                left_data, center_data, right_data, opt_for_basis_changle=True
            )
            if swappable_check:
                # print("at index ", i)
                results.add(i)
                remove_index_set.add(i + 1)
                remove_index_set.add(i + 2)
    remove_index_max = 0
    for idx in results:
        if idx > remove_index_max:
            # print("idx", idx)
            pauli_a = pauli_bit_groups[idx][0]
            pauli_b, pauli_c = pauli_bit_groups[idx + 1]
            pauli_b_str = pauli_b.get_pauli_str()
            pauli_c_str = pauli_c.get_pauli_str()
            pauli_d = pauli_bit_groups[idx + 2][0]
            # idx-1, idx+3 layerの両方にpauli_bが存在するか
            cand_str_left = [elem.get_pauli_str() for elem in pauli_bit_groups[idx - 1]]
            cand_str_right = [
                elem.get_pauli_str() for elem in pauli_bit_groups[idx + 3]
            ]
            if pauli_b_str in cand_str_left and pauli_b_str in cand_str_right:
                print(f"Bが両方に存在するようです: {pauli_b_str}")
                print(pauli_bit_groups[idx - 1])
                print(pauli_bit_groups[idx + 3])
                remove_index_max = idx + 4
            # idx-1, idx+3 layerの両方にpauli_cが存在するか
            elif pauli_c_str in cand_str_left and pauli_c_str in cand_str_right:
                print(f"Cが両方に存在するようです: {pauli_c_str}")
                print(pauli_bit_groups[idx - 1])
                print(pauli_bit_groups[idx + 3])
                remove_index_max = idx + 4
        else:
            print("skip index", idx)
    return pauli_bit_groups


def three_layer_nontrivial_swap(pauli_bit_groups, with_mcr_index=False):
    remove_index_set = set()
    results = set()
    for i in range(len(pauli_bit_groups) - 2):
        if i not in remove_index_set:
            left_data = pauli_bit_groups[i]
            center_data = pauli_bit_groups[i + 1]
            right_data = pauli_bit_groups[i + 2]
            swappable_check = find_nontrivial_swap(left_data, center_data, right_data)
            if swappable_check:
                # print("at index ", i)
                results.add(i)
                pauli_bit_groups[i] = swappable_check[0]
                pauli_bit_groups[i + 2] = swappable_check[2]
                if len(grouping(pauli_bit_groups[i + 2])) >= 2:
                    remove_index_set.add(i + 2)
    # print("After 3-layer swap:", sum(pauli_bit_groups, []))
    if with_mcr_index:
        return sum(pauli_bit_groups, []), results
    return sum(pauli_bit_groups, [])


def optimize_data_loop(pauli_bit_lst, max_attempts=1, show_opt_log=False):
    clifford_lst = []
    initial = deepcopy(pauli_bit_lst)
    # clifford, data = loop_optimization(pauli_bit_lst, show_log=False)
    # clifford_lst.extend(clifford)
    # PauliBitのリストが2重の場合は、groupingを行わない。
    if contains_list(pauli_bit_lst):
        skip_grouping = True
        data = deepcopy(sum(pauli_bit_lst, []))
    else:
        skip_grouping = False
        data = deepcopy(pauli_bit_lst)
    tmp1 = deepcopy(data)

    attempts_left = max_attempts
    current_length = len(data)
    iteration = 1
    # print(f"☑️ current length: {current_length} gates")
    if not skip_grouping:
        data = three_layer_nontrivial_swap(grouping(data))
        clifford_1, data = loop_optimization(data, show_log=False)
        clifford_lst.extend(clifford_1)
    if len(data) == 0:
        if show_opt_log:
            print(f"🎉 Optimization success: {current_length} → {len(data)}")
        return clifford_lst, data
    # print(f"☑️ Initial optimization: {len(data)} gates")

    # while attempts_left > 0 and current_length > 0:
    while attempts_left > 0 and current_length > 1:  # 一時的に変更!!
        original_data = deepcopy(data)
        if (
            skip_grouping and iteration == 1
        ):  # あえてIdentityを挿入しているケースの場合は初回だけgroupingを行わない
            mcr_swapped_data = mcr_swap(pauli_bit_lst)
        else:
            mcr_swapped_data = mcr_swap(grouping(data))
        clifford_2, data = loop_optimization(mcr_swapped_data, show_log=False)
        # assert equiv([[], mcr_swapped_data], [clifford_2, data]), (
        #     f"equiv_loop_optimization: {mcr_swapped_data} != {clifford_2} + {data}"
        # )
        # print("生成されたCliffordゲート数:", len(clifford_2))
        clifford_lst.extend(clifford_2)
        # print("Clifford_lstにあるCliffordゲート数:", len(clifford_lst))

        if len(data) >= current_length:
            attempts_left -= 1
            # 一応equiv check
            # assert equiv([[], tmp1], [clifford_lst, data]), (
            #     f"equiv failed in 最適化できなかったとき {iteration}:\n{tmp1} != \n{clifford_lst} \n + {data}"
            # )
            if show_opt_log:
                print(
                    f"🔍 No optimization in iteration {iteration}: {current_length} → {len(data)}"
                )
                if original_data == data:
                    print("No data change.")
        else:
            if show_opt_log:
                print(
                    f"🎉 Optimization success in iteration {iteration}: {current_length} → {len(data)}"
                )
            # # 一応equiv check
            # assert equiv([[], tmp1], [clifford_lst, data]), (
            #     f"equiv failed in 最適化できたとき {iteration}: {tmp1} != {clifford_lst} + {data}"
            # )
            current_length = len(data)
            iteration += 1
    # assert equiv([[], tmp1], [clifford_lst, data]), (
    #     f"optimize_data_loop failed!"
    # )  # ここでエラー
    return clifford_lst, data


def attempt_mcr_retry(non_clifford_pauli_lst, reverse_input=False):
    # MCRを満たすゲートをあえて挿入する。

    grouped_data = grouping(non_clifford_pauli_lst, reverse_input=reverse_input)

    # print("⚠️ Trying to improve further with MCR identity insertion...")

    for idx, group in enumerate(grouped_data[:-1]):
        # if len(group) != 1 or len(grouped_data[idx + 1]) != 2:
        if len(grouped_data[idx + 1]) != 2:
            continue
        pauli_a = group[0]
        # pauli_d_str = grouped_data[idx + 2][0].get_pauli_str()
        pauli_b, pauli_c = grouped_data[idx + 1]
        new_pauli_str = multiply_all([pauli_a, pauli_b, pauli_c])[1]
        # cand_strs = [pauli_bit.get_pauli_str() for pauli_bit in grouped_data[idx + 2]]
        # if pauli_a.get_pauli_str() not in cand_strs and new_pauli_str not in cand_strs:
        # print("スキップ: ", grouped_data[idx + 1])
        # continue
        # print(f"add {new_pauli_str} at index {idx}")
        grouped_data[idx] += [
            PauliBit(new_pauli_str, np.pi / 4),
            PauliBit(new_pauli_str, -np.pi / 4),
        ]
    return grouped_data


def full_optimization(data, max_iter=3, show_opt_log=False, skip_additional_opt=False):
    final_clifford_lst = []
    initial = deepcopy(data)
    if isinstance(data, PauliRotationSequence):
        result = []
        for elem in data.get_all():
            sgn = str(elem[1])[0]
            pauli_str = str(elem[1])[1:]
            if sgn == "+":
                result.append(PauliBit(pauli_str, np.pi / 4))
            else:
                assert sgn == "-", f"Unexpected sign: {sgn}"
                result.append(PauliBit(pauli_str, -np.pi / 4))
        data = result
    for k in range(max_iter):
        if show_opt_log:
            print(f"🔁 Optimization iteration: {k + 1} / {max_iter}")
        clifford_lst, optimized_data = optimize_data_loop(
            data, show_opt_log=show_opt_log
        )
        final_clifford_lst.extend(clifford_lst)

        # if len(optimized_data) == 0:
        if len(optimized_data) <= 1:
            return final_clifford_lst, optimized_data

        if skip_additional_opt:
            final_clifford_lst.extend(clifford_lst)
            data = deepcopy(optimized_data)  # update data for next iteration
            new_optimized_data = data
            continue

        if show_opt_log:
            print(f"⚙️  Additional optimization: {k + 1} / {max_iter}")
        old_optimized_data = deepcopy(optimized_data)
        redundant_data = attempt_mcr_retry(optimized_data)
        new_clifford_lst, new_optimized_data = optimize_data_loop(
            redundant_data, show_opt_log=show_opt_log, max_attempts=1
        )

        # if len(new_optimized_data) == 0:
        if len(new_optimized_data) <= 1:
            final_clifford_lst.extend(new_clifford_lst)
            return final_clifford_lst, new_optimized_data
        else:
            if len(new_optimized_data) <= len(old_optimized_data):
                final_clifford_lst.extend(new_clifford_lst)
                data = deepcopy(new_optimized_data)  # update data for next iteration
            else:
                if show_opt_log:
                    print(
                        f"❗️ Additional optimization did not improve the result, stopping: {k + 1} / {max_iter}"
                    )
                return final_clifford_lst, old_optimized_data
    return final_clifford_lst, new_optimized_data


def main():
    filetype = "seq"
    nqubits = 20
    max_iter = 3

    with open(f"unopt_{nqubits}.pickle", "rb") as f:
        seq = pickle.load(f)

    data = []
    for elem in seq:
        sign = str(elem[1])[0]
        pauli_str = str(elem[1])[1:]
        angle = np.pi / 4 if sign == "+" else -np.pi / 4
        data.append(PauliBit(pauli_str, angle))

    data.append(PauliBit("Z" * nqubits, -np.pi / 4))  # Identity gate

    st = time()
    clifford_lst, optimized_data = full_optimization(
        data, max_iter=max_iter, show_opt_log=True, skip_additional_opt=False
    )
    ed = time()
    print(f"✅ Optimization completed in {ed - st:.5f} seconds")
    print(f"Final non-Clifford gates: {len(optimized_data)}")

    if nqubits <= 3:
        circuit_input = QuantumCircuit(nqubits)
        circuit_input.merge_circuit(
            PauliBit("Z" * nqubits, np.pi / 4).convert_into_qulacs()
        )
        circuit_input.merge_circuit(
            PauliBit("Z" * nqubits, -np.pi / 4).convert_into_qulacs()
        )

        circuit_output = QuantumCircuit(nqubits)
        circuit_output = set_clifford_to_qulacs(circuit_output, clifford_lst)
        if len(optimized_data) > 0:
            for elem in optimized_data:
                circuit_output.merge_circuit(elem.convert_into_qulacs())

        assert equivalence_check_via_mqt_qcec(
            circuit_input, circuit_output, exclude_zx_checker=True
        )


if __name__ == "__main__":
    main()
