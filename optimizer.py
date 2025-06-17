# three_layer_nontrivial_swapを実行することはゲートを追加挿入してmcr_swapを実行することと同じ？
from mcr.gate_apply import PauliBit, grouping, loop_optimization, multiply_all
from mcr.mcr_optimize import find_mcr, find_nontrivial_swap
from mcr.equiv_check import (
    equiv,
    equivalence_check_via_mqt_qcec,
)
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


def optimize_data_loop(pauli_bit_lst, max_attempts=1, show_opt_log=False):
    clifford_lst = []
    # print(f"🔁 Initial optimization: {current_length} gates")
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
    while attempts_left > 0 and current_length > 0:
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


def attempt_mcr_retry(non_clifford_pauli_lst):
    # MCRを満たすゲートをあえて挿入する。

    grouped_data = grouping(non_clifford_pauli_lst)

    # print("⚠️ Trying to improve further with MCR identity insertion...")

    for idx, group in enumerate(grouped_data[:-1]):
        if len(group) != 1 or len(grouped_data[idx + 1]) != 2:
            continue
        pauli_a = group[0]
        # pauli_d = grouped_data[idx + 2][0]
        pauli_b, pauli_c = grouped_data[idx + 1]
        new_pauli_str = multiply_all([pauli_a, pauli_b, pauli_c])[1]
        grouped_data[idx] += [
            PauliBit(new_pauli_str, np.pi / 4),
            PauliBit(new_pauli_str, -np.pi / 4),
        ]
    return grouped_data


def full_optimization(data, max_iter=10, show_opt_log=False, skip_additional_opt=False):
    final_clifford_lst = []
    initial = deepcopy(data)
    for k in range(max_iter):
        if show_opt_log:
            print(f"🔁 Optimization iteration: {k + 1} / {max_iter}")
        clifford_lst, optimized_data = optimize_data_loop(
            data, show_opt_log=show_opt_log
        )
        final_clifford_lst.extend(clifford_lst)

        if len(optimized_data) == 0:
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

        if len(new_optimized_data) == 0:
            final_clifford_lst.extend(new_clifford_lst)
            return final_clifford_lst, new_optimized_data
        else:
            if len(new_optimized_data) <= len(old_optimized_data):
                final_clifford_lst.extend(new_clifford_lst)
                data = deepcopy(new_optimized_data)  # update data for next iteration
            else:
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
