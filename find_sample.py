from mcr.gate_apply import PauliBit
from mcr.mcr_optimize import find_mcr, find_nontrivial_swap
from mcr.gate_apply import grouping, loop_optimization
from mcr.equiv_check import pauli_bit_equivalence_check, equivalence_check_via_mqt_qcec
from qulacs import QuantumCircuit
from mcr.gate_apply import set_clifford_to_qulacs
from time import time
import numpy as np
import pickle
from mcr.rotation_circuit import PauliRotationSequence
from mcr.unoptimize import unoptimize_circuit
from tqdm import tqdm


def mcr_swap(pauli_bit_groups):
    removed_group_indices = set()
    counter = 0
    for i in range(len(pauli_bit_groups) - 1):
        if i not in removed_group_indices:
            left_data = pauli_bit_groups[i]
            right_data = pauli_bit_groups[i + 1]
            sols = find_mcr(left_data, right_data)
            if sols:
                for gates in sols:
                    gate_a, gate_b, gate_c, gate_d = [gate[1:] for gate in gates]
                    # print(
                    # f"MCR swap!: {i}, {i + 1} -> {gate_a}, {gate_b}, {gate_c}, {gate_d}"
                    # )
                    left_data_for_swap, reduced_left_data = [], []
                    for ele in left_data:
                        if ele.get_pauli_str() in [gate_a, gate_b]:
                            left_data_for_swap.append(ele)
                        else:
                            reduced_left_data.append(ele)
                    assert len(left_data_for_swap) == 2, (
                        f"Expected 2 elements for swap, got {len(left_data_for_swap)}"
                    )

                    right_data_for_swap, reduced_right_data = [], []
                    for ele in right_data:
                        if ele.get_pauli_str() in [gate_c, gate_d]:
                            right_data_for_swap.append(ele)
                        else:
                            reduced_right_data.append(ele)

                    assert len(right_data_for_swap) == 2, (
                        f"Expected 2 elements for swap, got {len(right_data_for_swap)}"
                    )
                    pauli_bit_groups[i] = reduced_left_data + right_data_for_swap
                    pauli_bit_groups[i + 1] = left_data_for_swap + reduced_right_data
                    removed_group_indices.add(i + 1)
                    removed_group_indices.add(i + 2)  # もしかしてここが重要!?
                    counter += 1
                    break
    new_data = sum(pauli_bit_groups, [])
    # print(f"Total MCR swaps made: {counter}")
    return new_data


def three_layer_nontrivial_swap(pauli_bit_groups):
    initial = pauli_bit_groups.copy()
    removed_group_indices = set()
    for i in range(len(pauli_bit_groups) - 2):
        if i not in removed_group_indices:
            left_data = pauli_bit_groups[i]
            center_data = pauli_bit_groups[i + 1]
            right_data = pauli_bit_groups[i + 2]
            swappable_check = find_nontrivial_swap(left_data, center_data, right_data)
            if swappable_check:
                pauli_bit_groups[i] = swappable_check[0]
                pauli_bit_groups[i + 2] = swappable_check[2]

                removed_group_indices.add(i + 1)
                removed_group_indices.add(i + 2)
    new_data = sum(pauli_bit_groups, [])
    return new_data


def test_algorithm(pauli_bit_lst, show_opt_log=True):
    clifford_lst = []
    clifford, data_for_optimization = loop_optimization(pauli_bit_lst, show_log=False)
    clifford_lst.extend(clifford)

    flag = True
    length = len(data_for_optimization)
    while flag and len(data_for_optimization) > 0:
        groups = grouping(data_for_optimization)
        # groupingした後にfind_nontrivial_swapを適用し、loop_optimizationを行う
        swapped_new_data = three_layer_nontrivial_swap(groups)
        clifford_1, data_for_optimization = loop_optimization(
            swapped_new_data, show_log=False
        )

        new_data = mcr_swap(grouping(data_for_optimization))
        clifford_2, data_for_optimization = loop_optimization(new_data, show_log=False)
        clifford_lst.extend(clifford_1)
        clifford_lst.extend(clifford_2)
        if len(data_for_optimization) >= length:
            flag = False
        else:
            if show_opt_log:
                print(
                    f"🎉 Successful optimization using MCR! {length} -> {len(data_for_optimization)}"
                )
            length = len(data_for_optimization)
            if length == 0:
                flag = False
    return clifford_lst, data_for_optimization


def main():
    filetype = "seq"  # "small" or "seq"
    num_samples = 100
    nqubits = 2
    with_swap_option = True  # If True, the MCR swap is executed (then the unoptimized circuit becomes longer)
    # Number of iterations for the unoptimized circuit
    unopt_iteration_count = 2
    for _ in tqdm(
        range(num_samples), desc=f"Processing {filetype} circuits with {nqubits} qubits"
    ):
        # gen unopt circuit
        input_seq = PauliRotationSequence(nqubits)
        initial_pauli_string = "Z" * nqubits
        input_seq.add_gate((0,), f"+{initial_pauli_string}")
        # duplicate the circuit
        # Perform unoptimization
        unopt_seq = unoptimize_circuit(
            input_seq, unopt_iteration_count, with_swap_option
        )
        with open(f"unopt_{nqubits}.pickle", "wb") as f:
            pickle.dump(unopt_seq.get_all(), f)
        with open(f"unopt_{nqubits}.pickle", "rb") as f:
            seq = pickle.load(f)
        seq = unopt_seq.get_all()

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
        clifford_lst, optimized_data = test_algorithm(data)
        ed = time()
        if len(optimized_data) > 0:
            print("found!!!", len(optimized_data), "gates")
            break
        # print(f"Optimization time: {ed - st} seconds")
        # for elem in grouping(optimized_data):
        #     print(f"Group size: {len(elem)}")
        #     print(elem)
        #     print("==========================")

        # if nqubits <= 6:
        #     circuit_input = QuantumCircuit(nqubits)
        #     for elem in data:
        #         circuit_input.merge_circuit(elem.convert_into_qulacs())

        #     circuit_output = QuantumCircuit(nqubits)
        #     circuit_output = set_clifford_to_qulacs(circuit_output, clifford_lst)
        #     for elem in optimized_data:
        #         circuit_output.merge_circuit(elem.convert_into_qulacs())

        #     equivalence_check_via_mqt_qcec(
        #         circuit_input, circuit_output, exclude_zx_checker=True
        #     )


if __name__ == "__main__":
    main()
