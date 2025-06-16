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


def mcr_swap(pauli_bit_groups, with_mcr_index=False):
    data = deepcopy(pauli_bit_groups)
    results = set()
    remove_index_set = set()
    for i in range(len(data) - 1):
        if i not in remove_index_set:
            left_data = data[i]
            right_data = data[i + 1]
            swappable_check = find_mcr(left_data, right_data)
            if swappable_check:
                # print(f"Swapping {left_data} and {right_data} at index {i}")
                # assert equiv(
                #     [[], left_data + right_data],
                #     [[], swappable_check[0] + swappable_check[1]],
                # ), (
                #     f"Swapping failed: {left_data} + {right_data} != {swappable_check[0]} + {swappable_check[1]}"
                # )
                results.add(i)
                data[i] = swappable_check[0]
                data[i + 1] = swappable_check[1]
                if len(grouping(data[i + 1])) >= 2:
                    remove_index_set.add(i + 1)
    new_data = sum(data, [])
    # assert equiv([[], sum(data, [])], [[], new_data]), (
    #     f"MCR swapで失敗しています！: {data} → {new_data}"
    # )
    if with_mcr_index:
        return new_data, results
    return new_data


def optimize_data_loop(pauli_bit_lst, max_attempts=1, show_opt_log=False):
    clifford_lst = []
    # print(f"🔁 Initial optimization: {current_length} gates")
    # clifford, data = loop_optimization(pauli_bit_lst, show_log=False)
    # clifford_lst.extend(clifford)

    attempts_left = max_attempts
    data = deepcopy(pauli_bit_lst)
    current_length = len(data)
    iteration = 1
    while attempts_left > 0 and current_length > 0:
        original_data = deepcopy(data)

        new_data = mcr_swap(grouping(data))
        clifford_2, data = loop_optimization(new_data, show_log=False)
        clifford_lst.extend(clifford_2)

        if len(data) >= current_length:
            attempts_left -= 1
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
            current_length = len(data)
            iteration += 1
    return clifford_lst, data


def attempt_mcr_retry(clifford_lst, optimized_data):
    initial_clifford_data = deepcopy(clifford_lst)
    # initial_data = deepcopy(optimized_data)

    grouped_data = grouping(optimized_data)

    # print("⚠️ Trying to improve further with MCR identity insertion...")

    for idx, group in enumerate(grouped_data[:-1]):
        if len(group) != 1 or len(grouped_data[idx + 1]) != 2:
            continue
        pauli_a = group[0]
        # pauli_d = grouped_data[idx + 2][0]
        pauli_b, pauli_c = grouped_data[idx + 1]
        new_pauli_str = multiply_all([pauli_a, pauli_b, pauli_c])[1]
        # print(f"Inserting new PauliBit with string {new_pauli_str} at index {idx}")
        grouped_data[idx] += [
            PauliBit(new_pauli_str, np.pi / 4),
            PauliBit(new_pauli_str, -np.pi / 4),
        ]
    target_data = mcr_swap(grouped_data)
    # assert equiv([[], sum(t, [])], [[], target_data]), "MCR swap failed!"
    additional_clifford, new_optimized_data = optimize_data_loop(
        target_data, show_opt_log=True, max_attempts=1
    )

    # return clifford_lst, optimized_data
    updated_clifford_lst = initial_clifford_data + additional_clifford
    return updated_clifford_lst, new_optimized_data


def full_optimization(data, max_iter=10, show_opt_log=False):
    clifford_lst, optimized_data = optimize_data_loop(data, show_opt_log=show_opt_log)
    # assert equiv([[], data], [clifford_lst, optimized_data]), (
    #     "First optimization failed"
    # )
    if len(optimized_data) == 0:
        return clifford_lst, optimized_data

    for k in range(max_iter):
        if show_opt_log:
            print(f"🔁 Additional optimization: {k + 1} / {max_iter}")
        # initial_optimized_data = deepcopy(optimized_data)
        # initial_clifford_lst = deepcopy(clifford_lst)
        new_clifford_lst, new_optimized_data = attempt_mcr_retry(
            clifford_lst, optimized_data
        )
        # assert equiv(
        #     [initial_clifford_lst, initial_optimized_data],
        #     [new_clifford_lst, new_optimized_data],
        # ), "Optimization failed to maintain equivalence"
        if len(new_optimized_data) == 0:
            print("OK!")
            clifford_lst = new_clifford_lst
            optimized_data = new_optimized_data
            return clifford_lst, optimized_data
        else:
            if len(new_optimized_data) <= len(optimized_data):
                clifford_lst = deepcopy(new_clifford_lst)
                optimized_data = deepcopy(new_optimized_data)
            else:
                print(
                    f"❗️ Additional optimization did not improve the result, stopping: {k + 1} / {max_iter}"
                )
                return clifford_lst, optimized_data
    return clifford_lst, optimized_data


def main():
    filetype = "seq"
    nqubits = 2
    max_iter = 10

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
        data, max_iter=max_iter, show_opt_log=True
    )
    ed = time()
    print(f"✅ Optimization completed in {ed - st:.5f} seconds")

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
