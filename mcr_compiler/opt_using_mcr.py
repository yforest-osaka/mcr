# 廃止予定
from mcr.gate_apply import PauliBit
from mcr.mcr_optimize import find_mcr, find_nontrivial_swap
from mcr.gate_apply import grouping, loop_optimization, multiply_all
from mcr.equiv_check import (
    pauli_bit_equivalence_check,
    equivalence_check_via_mqt_qcec,
    equiv,
)
from qulacs import QuantumCircuit
from mcr.gate_apply import set_clifford_to_qulacs
from mcr.rotation_circuit import PauliRotationSequence
from time import time
import numpy as np
import pickle
from copy import deepcopy
import random


def mcr_swap(pauli_bit_groups, with_mcr_index=False):
    initial = deepcopy(pauli_bit_groups)
    results = set()
    counter = 0
    remove_index_set = set()
    for i in range(len(pauli_bit_groups) - 1):
        if i not in remove_index_set:
            left_data = pauli_bit_groups[i]
            right_data = pauli_bit_groups[i + 1]
            swappable_check = find_mcr(left_data, right_data)
            # print("-----------------")
            # print(left_data, right_data)
            if swappable_check:
                results.add(i)
                # print(f"MCR Swappable found!: {i}")
                # print(swappable_check)
                # print("-----------------")
                pauli_bit_groups[i] = swappable_check[0]
                pauli_bit_groups[i + 1] = swappable_check[1]
                group_count = grouping(pauli_bit_groups[i + 1])
                if len(group_count) >= 2:
                    remove_index_set.add(i + 1)
                counter += 1
    new_data = sum(pauli_bit_groups, [])
    # assert equiv([[], sum(initial, [])], [[], new_data]), "MCR swap failed!"
    if with_mcr_index:
        return new_data, results
    return new_data


def three_layer_nontrivial_swap(pauli_bit_groups):
    initial = deepcopy(pauli_bit_groups)
    counter = 0
    remove_index_set = set()
    for i in range(len(pauli_bit_groups) - 2):
        if i not in remove_index_set:
            left_data = pauli_bit_groups[i]
            center_data = pauli_bit_groups[i + 1]
            right_data = pauli_bit_groups[i + 2]
            # print(left_data, center_data, right_data)
            swappable_check = find_nontrivial_swap(left_data, center_data, right_data)
            if swappable_check:
                # print(f"3 layer Swappable found!: {i}")
                # print(swappable_check)
                # print("-----------------")
                pauli_bit_groups[i] = swappable_check[0]
                pauli_bit_groups[i + 2] = swappable_check[2]
                # remove_index_set.add(i + 1)
                group_count = grouping(pauli_bit_groups[i + 2])
                if len(group_count) >= 2:
                    remove_index_set.add(i + 2)
                counter += 1
            # if counter == 1:
            #     break
    new_data = sum(pauli_bit_groups, [])
    # assert equiv([[], sum(initial, [])], [[], new_data]), "Three-layer swap failed!"
    return new_data


def test_algorithm(
    pauli_bit_lst, show_opt_log=True, force_order_of_grouping=None, flag_trial=20
):
    if isinstance(pauli_bit_lst, PauliRotationSequence):
        data = []
        for elem in pauli_bit_lst.get_all():
            sgn = str(elem[1])[0]
            pauli_str = str(elem[1])[1:]
            if sgn == "+":
                data.append(PauliBit(pauli_str, np.pi / 4))
            else:
                assert sgn == "-", f"Unexpected sign: {sgn}"
                data.append(PauliBit(pauli_str, -np.pi / 4))
        pauli_bit_lst = data
    clifford_lst = []
    clifford, data_for_optimization = loop_optimization(pauli_bit_lst, show_log=False)
    clifford_lst.extend(clifford)
    flag = flag_trial  # 試行回数
    mcr_flag = flag
    length = len(data_for_optimization)
    k = 1
    while flag > 0 and length > 0:
        order_of_grouping = random.choice(["left", "right"])
        if force_order_of_grouping is not None:
            order_of_grouping = force_order_of_grouping
        # print("Using order_of_grouping:", order_of_grouping)
        # print(k, "th iteration")
        # print("flag_value:", flag)
        # print("mcr_flag_value:", mcr_flag)
        initial = deepcopy(data_for_optimization)
        if order_of_grouping == "left":
            groups = grouping(data_for_optimization, reverse_input=False)
        else:
            groups = grouping(data_for_optimization, reverse_input=True)
        # groupingした後にfind_nontrivial_swapを適用し、loop_optimizationを行う
        swapped_new_data = three_layer_nontrivial_swap(groups)
        # assert equiv([[], initial], [[], swapped_new_data]), "INITIAL Swap failed!"
        # print("swapped_new_data:\n", swapped_new_data)
        clifford_1, data_for_optimization = loop_optimization(
            swapped_new_data, show_log=False
        )
        # if len(data_for_optimization) != 0:
        #     assert equiv([[], swapped_new_data], [clifford_1, data_for_optimization]), (
        #         "Swap failed!"
        #     )
        # print(f"Length after swap: {len(data_for_optimization)}")
        # print(data_for_optimization)
        # print("-----------------")
        if len(data_for_optimization) == 0:
            # print("No data left after swap!")
            clifford_lst.extend(clifford_1)
            flag = 0
            mcr_flag = 0

        if mcr_flag > 0:
            if order_of_grouping == "left":
                new_data = mcr_swap(
                    grouping(data_for_optimization, reverse_input=False)
                )
            else:
                new_data = mcr_swap(grouping(data_for_optimization, reverse_input=True))
            clifford_2, data_for_optimization = loop_optimization(
                new_data, show_log=False
            )
            print(f"Length after mcrswap: {len(data_for_optimization)}")
            # print(data_for_optimization)
            clifford_lst.extend(clifford_1)
            clifford_lst.extend(clifford_2)
        # print(f"Length after MCR swap: {len(data_for_optimization)}, {length}")
        if len(data_for_optimization) >= length:
            flag -= 1
            mcr_flag -= 1
            if show_opt_log:
                print(
                    f"🔍 No optimization found in {k}th iteration. Try {mcr_flag + 1} times left... {length} -> {len(data_for_optimization)}"
                )
                if initial == data_for_optimization:
                    print("No change in data after optimization!")
        else:
            if show_opt_log:
                print(
                    f"🎉 Successful optimization using MCR! {length} -> {len(data_for_optimization)}"
                )
            length = len(data_for_optimization)
            k += 1
            # if length == 0:
            if length <= 1:
                flag = 0
    return clifford_lst, data_for_optimization


def main():
    filetype = "seq"  # "small" or "seq"
    nqubits = 2
    # with open(f"unopt_{filetype}.pickle", "rb") as f:
    with open(f"unopt_{nqubits}.pickle", "rb") as f:
        seq = pickle.load(f)
    data = []
    for elem in seq:
        sgn = str(elem[1])[0]
        pauli_str = str(elem[1])[1:]
        if sgn == "+":
            data.append(PauliBit(pauli_str, np.pi / 4))
        else:
            assert sgn == "-", f"Unexpected sign: {sgn}"
            data.append(PauliBit(pauli_str, -np.pi / 4))
    data.append(PauliBit("Z" * nqubits, -np.pi / 4))  # Add identity gate

    st = time()
    clifford_lst, optimized_data = test_algorithm(data, show_opt_log=True)

    assert equiv([[], data], [clifford_lst, optimized_data]), "Test algorithm failed!"

    # MCRが存在する場合は少しだけIdentityを挿入して再度アルゴリズムを実行させる
    grouped_optimized_data = grouping(optimized_data)
    aft_mcr, mcr_indices = mcr_swap(grouped_optimized_data, with_mcr_index=True)
    # aft_mcr_2 = three_layer_nontrivial_swap(grouped_optimized_data)
    if aft_mcr != optimized_data:
        print("Try!")
        for idx in mcr_indices:
            pauli_a = grouped_optimized_data[idx - 1][0]
            if len(grouped_optimized_data[idx]) == 2:
                pauli_b, pauli_c = grouped_optimized_data[idx]
                pauli_str_for_insert = multiply_all([pauli_a, pauli_b, pauli_c])[1]
                grouped_optimized_data.insert(
                    idx + 1,
                    [
                        PauliBit(pauli_str_for_insert, np.pi / 4),
                        PauliBit(pauli_str_for_insert, -np.pi / 4),
                    ],
                )
            else:
                print(
                    "要素数が2個でないため挿入をスキップしました: ",
                    grouped_optimized_data[idx],
                )
        assert equiv([[], aft_mcr], [[], sum(grouped_optimized_data, [])]), (
            "MCR addition failed!"
        )
        additional_clifford, optimized_data = test_algorithm(
            three_layer_nontrivial_swap(grouped_optimized_data)
        )
        clifford_lst.extend(additional_clifford)

    ed = time()
    print(f"Optimization time: {ed - st} seconds")

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
        # for elem in grouping(optimized_data):
        # print(elem)
        for elem in optimized_data:
            circuit_output.merge_circuit(elem.convert_into_qulacs())

        assert equivalence_check_via_mqt_qcec(
            circuit_input, circuit_output, exclude_zx_checker=True
        )


if __name__ == "__main__":
    main()
