# 廃止予定
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


def mcr_swap(pauli_bit_groups, with_mcr_index=False):
    initial = deepcopy(pauli_bit_groups)
    results = set()
    remove_index_set = set()
    for i in range(len(pauli_bit_groups) - 1):
        if i not in remove_index_set:
            left_data = pauli_bit_groups[i]
            right_data = pauli_bit_groups[i + 1]
            swappable_check = find_mcr(left_data, right_data)
            if swappable_check:
                print(f"Swapping {left_data} and {right_data} at index {i}")
                results.add(i)
                pauli_bit_groups[i] = swappable_check[0]
                pauli_bit_groups[i + 1] = swappable_check[1]
                if len(grouping(pauli_bit_groups[i + 1])) >= 2:
                    remove_index_set.add(i + 1)
    new_data = sum(pauli_bit_groups, [])
    if with_mcr_index:
        return new_data, results
    return new_data


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
                results.add(i)
                pauli_bit_groups[i] = swappable_check[0]
                pauli_bit_groups[i + 2] = swappable_check[2]
                if len(grouping(pauli_bit_groups[i + 2])) >= 2:
                    remove_index_set.add(i + 2)
    if with_mcr_index:
        return sum(pauli_bit_groups, []), results
    return sum(pauli_bit_groups, [])


def optimize_data_loop(pauli_bit_lst, max_attempts=2, show_opt_log=True):
    clifford_lst = []
    clifford, data = loop_optimization(pauli_bit_lst, show_log=False)
    clifford_lst.extend(clifford)

    attempts_left = max_attempts
    current_length = len(data)
    iteration = 1

    while attempts_left > 0 and current_length > 0:
        original_data = deepcopy(data)

        groups = grouping(data)
        swapped_data = three_layer_nontrivial_swap(groups)

        clifford_1, data = loop_optimization(swapped_data, show_log=False)
        clifford_lst.extend(clifford_1)

        if len(data) == 0:
            break

        new_data = mcr_swap(grouping(data))
        clifford_2, data = loop_optimization(new_data, show_log=False)
        clifford_lst.extend(clifford_2)

        if len(data) >= current_length:
            attempts_left -= 1
            if show_opt_log:
                print(
                    f"🔍 No optimization in {iteration}th iteration. {current_length} → {len(data)}"
                )
                if original_data == data:
                    print("No data change.")
        else:
            if show_opt_log:
                print(
                    f"🎉 Optimization success in iteration {iteration}: {current_length} → {len(data)}"
                )
            current_length = len(data)
            iteration += 1

    return clifford_lst, data


def attempt_mcr_retry(optimized_data, clifford_lst):
    grouped_data = grouping(optimized_data)
    aft_mcr, mcr_indices = mcr_swap(grouped_data, with_mcr_index=True)

    print("⚠️ Trying to improve further with MCR identity insertion...")

    for idx in mcr_indices:
        if idx == 0 or len(grouped_data[idx]) != 2:
            # print("スキップ: ", grouped_data[idx])
            continue
        pauli_a = grouped_data[idx - 1][0]
        # pauli_d = grouped_data[idx + 1][0]
        pauli_b, pauli_c = grouped_data[idx]
        new_pauli_str = multiply_all([pauli_a, pauli_b, pauli_c])[1]
        # new_pauli_str_2 = multiply_all([pauli_d, pauli_b, pauli_c])[1]

        grouped_data.insert(
            idx + 1,
            [
                PauliBit(new_pauli_str, np.pi / 4),
                PauliBit(new_pauli_str, -np.pi / 4),
            ],
        )

        # grouped_data.insert(
        #     idx + 1,
        #     [
        #         PauliBit(new_pauli_str_2, np.pi / 4),
        #         PauliBit(new_pauli_str_2, -np.pi / 4),
        #     ],
        # )

    assert equiv([[], aft_mcr], [[], sum(grouped_data, [])]), (
        "MCR identity insertion failed!"
    )

    additional_clifford, optimized_data = optimize_data_loop(
        three_layer_nontrivial_swap(grouped_data), show_opt_log=False
    )
    clifford_lst.extend(additional_clifford)

    return optimized_data, clifford_lst


def main():
    filetype = "seq"
    nqubits = 2

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
    clifford_lst, optimized_data = optimize_data_loop(data, show_opt_log=True)
    assert equiv([[], data], [clifford_lst, optimized_data]), (
        "First optimization failed"
    )

    max_iter = 50
    for k in range(max_iter):
        print(f"🔥 Additional optimization: {k + 1} / {max_iter}")
        optimized_data, clifford_lst = attempt_mcr_retry(optimized_data, clifford_lst)
        if len(optimized_data) == 0:
            print("🎉 All optimizations completed successfully!")
            break

    ed = time()
    print(f"✅ Optimization completed in {ed - st:.2f} seconds")
    print(optimized_data)

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
        for elem in optimized_data:
            circuit_output.merge_circuit(elem.convert_into_qulacs())

        assert equivalence_check_via_mqt_qcec(
            circuit_input, circuit_output, exclude_zx_checker=True
        )


if __name__ == "__main__":
    main()
