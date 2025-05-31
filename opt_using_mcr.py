from mcr.gate_apply import PauliBit
from mcr.mcr_optimize import find_mcr, find_nontrivial_swap
from mcr.gate_apply import grouping, loop_optimization
from mcr.equiv_check import (
    pauli_bit_equivalence_check,
    equivalence_check_via_mqt_qcec,
    equiv,
)
from qulacs import QuantumCircuit
from mcr.gate_apply import set_clifford_to_qulacs
from time import time
import numpy as np
import pickle
from copy import deepcopy


def mcr_swap(pauli_bit_groups):
    initial = deepcopy(pauli_bit_groups)
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
                remove_index_set.add(i + 1)
                group_count = grouping(pauli_bit_groups[i + 2])
                if len(group_count) >= 2:
                    remove_index_set.add(i + 2)
                counter += 1
            # if counter == 1:
            #     break
    new_data = sum(pauli_bit_groups, [])
    # assert equiv([[], sum(initial, [])], [[], new_data]), "Three-layer swap failed!"
    return new_data


def test_algorithm(pauli_bit_lst, show_opt_log=True):
    clifford_lst = []
    clifford, data_for_optimization = loop_optimization(pauli_bit_lst, show_log=False)
    clifford_lst.extend(clifford)
    flag = 100
    mcr_flag = flag
    length = len(data_for_optimization)
    k = 1
    while flag > 0 and length > 0:
        # print(k, "th iteration")
        # print("flag_value:", flag)
        # print("mcr_flag_value:", mcr_flag)
        initial = deepcopy(data_for_optimization)
        groups = grouping(data_for_optimization)
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
            new_data = mcr_swap(grouping(data_for_optimization))
            clifford_2, data_for_optimization = loop_optimization(
                new_data, show_log=False
            )
            # print(f"Length after swap: {len(data_for_optimization)}")
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
        else:
            if show_opt_log:
                print(
                    f"🎉 Successful optimization using MCR! {length} -> {len(data_for_optimization)}"
                )
            length = len(data_for_optimization)
            k += 1
            if length == 0:
                flag = 0

    #

    return clifford_lst, data_for_optimization


def main():
    filetype = "seq"  # "small" or "seq"
    nqubits = 3
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

        equivalence_check_via_mqt_qcec(
            circuit_input, circuit_output, exclude_zx_checker=True
        )


if __name__ == "__main__":
    main()
