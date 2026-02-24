import os
from copy import deepcopy
from uuid import uuid4
from typing import List, Tuple
import numpy as np
from mcr.gate_apply import (
    PauliBit,
    multiply_all,
)
from mcr.rotation_circuit import PauliRotationSequence
from mcr.filesave import qasm_to_pyzx
from mcr.optimize import optimize_process_pyzx


def find_mcr(
    left_bits: list[PauliBit],
    right_bits: list[PauliBit],
) -> list[tuple[str, str, str, str]]:
    length_left = len(left_bits)
    length_right = len(right_bits)
    for i in range(length_left - 1):
        for j in range(i + 1, length_left):
            pauli_A = left_bits[i]
            pauli_B = left_bits[j]

            for k in range(length_right - 1):
                pauli_C = right_bits[k]
                if any([pauli_A.commutes(pauli_C), pauli_B.commutes(pauli_C)]):
                    continue

                coef_abc, pat_abc = multiply_all([pauli_A, pauli_B, pauli_C])
                if abs(coef_abc.imag) > 1e-12:
                    continue
                if coef_abc == 1:
                    sgn = 0
                else:
                    assert coef_abc == -1
                    sgn = 1

                for l in range(k + 1, length_right):
                    target = right_bits[l]
                    target_sign = target.sgn
                    target_str = target.get_pauli_str()
                    if target_str == pat_abc:
                        if abs(pauli_A.get_angle()) != abs(pauli_B.get_angle()) or abs(
                            pauli_C.get_angle()
                        ) != abs(target.get_angle()):
                            continue

                        if target_sign == sgn:  # D = ABC
                            # 消してみる
                            # new_left = left_bits.copy()
                            # new_right = right_bits.copy()
                            # new_left.pop(j)
                            # new_left.pop(i)
                            # angle = pauli_A.get_angle()
                            # pat_A = pauli_A.get_pauli_str()
                            # additional_clifford = PauliBit(pat_A, 2 * angle)
                            # new_left += [additional_clifford, pauli_C, target]
                            # new_right.pop(l)
                            # new_right.pop(k)

                            # return (
                            #     new_left,
                            #     [
                            #         PauliBit(pat_A, -1 * angle),
                            #         pauli_B,
                            #     ]
                            #     + new_right,
                            # )
                            continue
                        else:
                            assert target_sign + sgn == 1, (
                                "Sign mismatch"
                            )  # D = -ABC (only swap)
                            new_left = left_bits.copy()
                            new_right = right_bits.copy()
                            new_left.pop(j)
                            new_left.pop(i)
                            new_left += [pauli_C, target]
                            new_right.pop(l)
                            new_right.pop(k)
                            return new_left, [pauli_A, pauli_B] + new_right
    return None


def find_nontrivial_swap(
    left_bits: list[PauliBit],
    center_bits: list[PauliBit],
    right_bits: list[PauliBit],
    opt_for_basis_changle=False,
):
    # ) -> tuple(list[PauliBit], list[PauliBit], list[PauliBit]):
    # check  A | B, C | D = D | B, C | A
    if len(center_bits) != 2:
        return None
    pauli_B, pauli_C = center_bits
    angle_b = pauli_B.get_angle()
    angle_c = pauli_C.get_angle()
    if abs(angle_b) != abs(angle_c):
        return None
    for l_idx, pauli_A in enumerate(left_bits):
        angle_a = pauli_A.get_angle()
        parity_lst = [pauli_A.commutes(pb) for pb in center_bits]  # {A, B}, {A, C}
        if any(parity_lst):
            continue
        coef_abc, pat_abc = multiply_all([pauli_A, pauli_B, pauli_C])

        if abs(coef_abc.imag) > 1e-12:
            continue

        if coef_abc == 1:
            sgn = 0
        else:
            assert coef_abc == -1
            sgn = 1
        for r_idx, pauli_D in enumerate(right_bits):
            target_str = pauli_D.get_pauli_str()
            angle_d = pauli_D.get_angle()
            if abs(angle_d) != abs(angle_a):
                continue
            if target_str == pat_abc:
                sign_d = pauli_D.sgn
                if sgn == sign_d:
                    if opt_for_basis_changle:
                        if (
                            len(left_bits) == 1
                            and len(center_bits) == 2
                            and len(right_bits) == 1
                        ):
                            return True
                        return False
                    left_bits.pop(l_idx)
                    right_bits.pop(r_idx)
                    return (
                        left_bits + [pauli_D],
                        center_bits,
                        [pauli_A] + right_bits,
                    )
                # 消してみる
                # else:
                #     assert sgn + sign_d == 1, "Sign mismatch"
                #     left_bits.pop(l_idx)
                #     right_bits.pop(r_idx)
                #     angle_a = pauli_A.get_angle()
                #     pat_A = pauli_A.get_pauli_str()
                #     new_left = left_bits + [
                #         PauliBit(pat_A, 2 * angle_a),
                #         pauli_D,
                #     ]
                #     new_right = [PauliBit(pat_A, -1 * angle_a)] + right_bits
                #     return new_left, center_bits, new_right
    return None


def grouping(pauli_bit_data_lst, reverse_input=False):
    L = []  # Empty list L

    if reverse_input:
        iterable = reversed(pauli_bit_data_lst)
    else:
        iterable = pauli_bit_data_lst

    for Rp in iterable:
        j = 0  # Initialize (create a new layer if no anti-commuting group is found)

        for k in reversed(range(len(L))):
            commute_info = [Rp.commutes(Rk) for Rk in L[k]]
            if not all(commute_info):  # If there is even one anti-commuting element
                j = k + 1
                break

        if j == 0:
            if len(L) == 0:
                L.append([Rp])
            else:
                L[0].append(Rp)
        else:
            if len(L) == j:
                # Create a new layer
                L.append([Rp])
            else:
                # Add to an existing group
                L[j].append(Rp)
    if reverse_input:
        # return list(reversed([list(reversed(layer)) for layer in L]))
        return list(reversed(L))
    return L


def synthesize_sequence(pauli_bit_data_lst):
    length = len(pauli_bit_data_lst)
    if length == 1:
        return pauli_bit_data_lst
    pauli_bit_data_lst = sorted(pauli_bit_data_lst, key=lambda x: x.get_pauli_str())
    results = []
    target_str = pauli_bit_data_lst[0].get_pauli_str()
    angle = pauli_bit_data_lst[0].get_angle()
    for idx, elem in enumerate(pauli_bit_data_lst[1:]):
        if target_str == elem.get_pauli_str():
            angle += elem.get_angle()
            if idx == length - 2 and angle != 0:  # last element must be added
                results.append(PauliBit(target_str, angle))
        else:
            if angle != 0:
                results.append(PauliBit(target_str, angle))
            target_str = elem.get_pauli_str()
            angle = elem.get_angle()
            if idx == length - 2:  # last element must be added
                results.append(PauliBit(target_str, angle))
    return results


def separate_clifford_and_rotation(pauli_bit_data_lst):
    clifford_gates = []
    rotation_gates = []
    for elem in pauli_bit_data_lst:
        if elem.is_clifford():
            clifford_gates += elem.get_clifford_gate_sequence()
        else:
            angle = elem.get_angle()
            angle = (angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
            if np.isclose(angle, np.pi) or np.isclose(angle, -np.pi):  # Pauli
                clifford_gates += PauliBit(
                    elem.get_pauli_str(), np.pi
                ).get_clifford_gate_sequence()
            elif np.isclose(angle, np.pi / 2):  # Clifford
                clifford_gates += PauliBit(
                    elem.get_pauli_str(), np.pi / 2
                ).get_clifford_gate_sequence()
            elif np.isclose(angle, -np.pi / 2):  # Clifford
                clifford_gates += PauliBit(
                    elem.get_pauli_str(), -np.pi / 2
                ).get_clifford_gate_sequence()
            elif np.isclose(angle, 0):  # Identity
                continue
            elif angle > np.pi / 2:  # Clifford + non-Clifford
                assert angle < np.pi, f"Angle {angle} exceeds expected range"
                residue_angle = angle - np.pi / 2
                clifford_gates += PauliBit(
                    elem.get_pauli_str(), np.pi / 2
                ).get_clifford_gate_sequence()
                rotation_gates.append(PauliBit(elem.get_pauli_str(), residue_angle))
            elif angle < -np.pi / 2:
                assert angle > -np.pi, f"Angle {angle} exceeds expected range"
                residue_angle = angle + np.pi / 2
                clifford_gates += PauliBit(
                    elem.get_pauli_str(), -np.pi / 2
                ).get_clifford_gate_sequence()
                rotation_gates.append(PauliBit(elem.get_pauli_str(), residue_angle))
            else:  # non-Clifford
                rotation_gates.append(elem)
            # if angle > np.pi or angle < -np.pi:
            #     if angle > 0:
            #         residue_angle = angle - np.pi
            #     else:
            #         residue_angle = angle + np.pi
            #     clifford_gates += PauliBit(
            #         elem.get_pauli_str(), np.pi
            #     ).get_clifford_gate_sequence()
            #     rotation_gates.append(PauliBit(elem.get_pauli_str(), residue_angle))
            # elif angle > np.pi / 2:
            #     assert angle < np.pi, f"Angle {angle} exceeds expected range"
            #     residue_angle = angle - np.pi / 2
            #     clifford_gates += PauliBit(
            #         elem.get_pauli_str(), np.pi / 2
            #     ).get_clifford_gate_sequence()
            #     rotation_gates.append(PauliBit(elem.get_pauli_str(), residue_angle))
            # elif angle < -np.pi / 2:
            #     assert angle > -np.pi, f"Angle {angle} exceeds expected range"
            #     residue_angle = angle + np.pi / 2
            #     clifford_gates += PauliBit(
            #         elem.get_pauli_str(), -np.pi / 2
            #     ).get_clifford_gate_sequence()
            #     rotation_gates.append(PauliBit(elem.get_pauli_str(), residue_angle))
            # else:
            #     rotation_gates.append(elem)
    return clifford_gates, rotation_gates


def get_rotation_relation(pauli_bit_1: PauliBit, pauli_bit_2: PauliBit):
    if pauli_bit_1.get_pauli_str() == pauli_bit_2.get_pauli_str():
        sum_angle = pauli_bit_1.get_angles() + pauli_bit_2.get_angles()
        modulus = int((sum_angle / (np.pi / 4)) % 8)
        if np.allclose(
            modulus, int(modulus)
        ):  # Check if modulus is an integer (if true, exactly decomposed into Clifford+T)
            return int(modulus)
        else:
            return "synthesize"
    elif pauli_bit_1.commutes(pauli_bit_2):
        return "commute"
    else:
        return "anti-commute"


def zhang_optimization(pauli_bit_data_lst: List[PauliBit]):
    rots = synthesize_sequence(pauli_bit_data_lst)
    clifford_gate_sequence, optimized_rotations = separate_clifford_and_rotation(rots)

    return optimized_rotations, clifford_gate_sequence


def loop_optimization(pauli_bit_list, show_log=True, reverse_input=False):
    flag = True
    length = len(pauli_bit_list)
    initial_length = length
    clifford_data = []
    k = 0
    while flag:
        updated_rots = []
        grouping_data = grouping(pauli_bit_list, reverse_input=reverse_input)
        non_clifford_rots_group, extracted_clifford_group = [], []
        for elem in grouping_data:
            tmp = zhang_optimization(elem)
            non_clifford_rots_group.append(tmp[0])
            extracted_clifford_group.append(tmp[1])
        if (
            sum(non_clifford_rots_group, []) != []
        ):  # Update non-Clifford gates to move Clifford gates outward
            for i in range(len(non_clifford_rots_group)):
                clifford_seq = extracted_clifford_group[i]
                if len(clifford_seq) > 0 and len(updated_rots) > 0:
                    updated_rots = [
                        ele.clifford_update(clifford_gate_sequence=clifford_seq)
                        for ele in updated_rots
                    ]
                    updated_rots += non_clifford_rots_group[i]
                    clifford_data += clifford_seq
                else:
                    updated_rots += non_clifford_rots_group[i]
                    clifford_data += clifford_seq
        else:  # case where the number of non-Clifford rotations becomes zero and only Clifford gates remain
            for clifford in extracted_clifford_group:
                clifford_data += clifford
        if len(updated_rots) < length:
            k += 1
            length = len(updated_rots)
            pauli_bit_list = updated_rots
        else:
            if show_log and k > 0:
                print("=" * 40)
                print(f"{k}-iteration optimization applied!")
                print(f"optimization result: {initial_length} -> {len(updated_rots)}")
                print("=" * 40)
            flag = False
    return clifford_data, updated_rots


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
                results.add(i)
                data[i] = swappable_check[0]
                data[i + 1] = swappable_check[1]
                if len(data[i + 1]) >= 3 and len(grouping(data[i + 1])) >= 2:
                    remove_index_set.add(i + 1)
                elif i < len(data) - 2 and have_common_pauli_str(
                    data[i + 1], data[i + 2]
                ):
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
                results.add(i)
                remove_index_set.add(i + 1)
                remove_index_set.add(i + 2)
    remove_index_max = 0
    for idx in results:
        if idx > remove_index_max:
            pauli_a = pauli_bit_groups[idx][0]
            pauli_b, pauli_c = pauli_bit_groups[idx + 1]
            pauli_b_str = pauli_b.get_pauli_str()
            pauli_c_str = pauli_c.get_pauli_str()
            pauli_d = pauli_bit_groups[idx + 2][0]
            cand_str_left = [elem.get_pauli_str() for elem in pauli_bit_groups[idx - 1]]
            cand_str_right = [
                elem.get_pauli_str() for elem in pauli_bit_groups[idx + 3]
            ]
            if pauli_b_str in cand_str_left and pauli_b_str in cand_str_right:
                remove_index_max = idx + 4
            elif pauli_c_str in cand_str_left and pauli_c_str in cand_str_right:
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
                results.add(i)
                pauli_bit_groups[i] = swappable_check[0]
                pauli_bit_groups[i + 2] = swappable_check[2]
                if len(grouping(pauli_bit_groups[i + 2])) >= 2:
                    remove_index_set.add(i + 2)
    if with_mcr_index:
        return sum(pauli_bit_groups, []), results
    return sum(pauli_bit_groups, [])


def optimize_data_loop(pauli_bit_lst, show_opt_log=False, reverse_input=False):
    clifford_lst = []
    if contains_list(pauli_bit_lst):
        skip_grouping = True
        data = deepcopy(sum(pauli_bit_lst, []))
    else:
        skip_grouping = False
        data = deepcopy(pauli_bit_lst)
    tmp1 = deepcopy(data)

    attempts_left = 3
    current_length = len(data)
    iteration = 1
    # if not skip_grouping:
    #     data = three_layer_nontrivial_swap(grouping(data, reverse_input=reverse_input))
    #     clifford_1, data = loop_optimization(
    #         data, show_log=False, reverse_input=reverse_input
    #     )
    #     clifford_lst.extend(clifford_1)

    # for _ in range(2 * max_attempts):
    while attempts_left > 0:
        original_data = deepcopy(data)
        if skip_grouping and iteration == 1:
            mcr_swapped_data = mcr_swap(pauli_bit_lst, show_log=False)
        else:
            mcr_swapped_data = mcr_swap(grouping(data))
        clifford_2, data = loop_optimization(mcr_swapped_data, show_log=False)
        clifford_lst.extend(clifford_2)
        data = three_layer_nontrivial_swap(grouping(data, reverse_input=reverse_input))
        clifford_1, data = loop_optimization(
            data, show_log=False, reverse_input=reverse_input
        )
        clifford_lst.extend(clifford_1)

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


def attempt_mcr_retry(non_clifford_pauli_lst, reverse_input=False):
    grouped_data = grouping(non_clifford_pauli_lst, reverse_input=reverse_input)

    for idx, group in enumerate(grouped_data[:-1]):
        if len(grouped_data[idx + 1]) != 2:
            continue
        pauli_a = group[0]
        pauli_b, pauli_c = grouped_data[idx + 1]
        new_pauli_str = multiply_all([pauli_a, pauli_b, pauli_c])[1]
        grouped_data[idx] += [
            PauliBit(new_pauli_str, np.pi / 4),
            PauliBit(new_pauli_str, -np.pi / 4),
        ]
    return grouped_data


def full_optimization(data, max_iter=1, show_opt_log=False):
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
    clifford_lst, optimized_data = optimize_data_loop(
        data, show_opt_log=show_opt_log, reverse_input=False
    )
    final_clifford_lst.extend(clifford_lst)

    if len(optimized_data) <= 1:  # Achieved optimality
        return final_clifford_lst, optimized_data
    old_optimized_data = deepcopy(optimized_data)
    redundant_data = attempt_mcr_retry(optimized_data, reverse_input=False)
    new_clifford_lst, new_optimized_data = optimize_data_loop(
        redundant_data,
        show_opt_log=show_opt_log,
        reverse_input=False,
    )

    if len(new_optimized_data) <= 1:  # Achieved optimality
        final_clifford_lst.extend(new_clifford_lst)
        return final_clifford_lst, new_optimized_data
    else:
        if len(new_optimized_data) <= len(old_optimized_data):
            final_clifford_lst.extend(new_clifford_lst)
            return final_clifford_lst, new_optimized_data
        else:
            return final_clifford_lst, old_optimized_data


def clifford_lst_to_qasm(clifford_lst: List[Tuple[str, Tuple[int]]], filepath: str):
    data_for_qasm = []
    max_qubit_index = -1
    for gate in clifford_lst:
        qubits = gate[1]
        max_qubit_index = max(max_qubit_index, max(qubits))
        if gate[0] == "H":
            data_for_qasm.append(f"h q[{qubits[0]}];")
        elif gate[0] == "S":
            data_for_qasm.append(f"s q[{qubits[0]}];")
        elif gate[0] == "Sdg":
            data_for_qasm.append(f"sdg q[{qubits[0]}];")
        elif gate[0] == "X":
            data_for_qasm.append(f"x q[{qubits[0]}];")
        elif gate[0] == "Y":
            data_for_qasm.append(f"y q[{qubits[0]}];")
        elif gate[0] == "Z":
            data_for_qasm.append(f"z q[{qubits[0]}];")
        elif gate[0] == "CNOT":
            data_for_qasm.append(f"cx q[{qubits[0]}],q[{qubits[1]}];")
        elif gate[0] == "CZ":
            data_for_qasm.append(f"cz q[{qubits[0]}],q[{qubits[1]}];")
        elif gate[0] == "SWAP":
            data_for_qasm.append(f"swap q[{qubits[0]}],q[{qubits[1]}];")
        else:
            raise ValueError(f"Unknown gate type: {gate}")
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{max_qubit_index + 1}];",
    ]
    with open(filepath, "w") as f:
        f.write("\n".join(header + data_for_qasm))


def clifford_compile(
    clifford_lst: List[Tuple[str, Tuple[int]]], with_file_save=True
) -> str:
    unique_id = str(uuid4())
    clifford_lst_to_qasm(clifford_lst, f"tmp/clifford_{unique_id}.qasm")
    pyzx_circuit = qasm_to_pyzx(f"tmp/clifford_{unique_id}.qasm")
    os.remove(f"tmp/clifford_{unique_id}.qasm")
    pyzx_circuit_opt = optimize_process_pyzx(pyzx_circuit)
    if with_file_save:
        with open(f"tmp/clifford_opt_{unique_id}.qasm", "w") as f:
            f.write(pyzx_circuit_opt.to_qasm())
    return pyzx_circuit_opt.to_qasm()


def extract_qasm(qasm: str) -> list[str]:
    parts = qasm.split(";")
    result = []
    found_qreg = False
    for part in parts:
        part = part.strip()
        if not part:
            continue
        stmt = part + ";"
        if found_qreg:
            result.append(stmt)
        elif part.startswith("qreg"):
            found_qreg = True
    return result


def output_opt_qasm_file(clifford_lst, non_clifford, nqubits, filepath):
    # Optimization for clifford part (using PyZX)
    header = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{nqubits}];"]
    if len(clifford_lst) > 0:
        qasm_opt_circuit = clifford_compile(clifford_lst, with_file_save=False)
        qasm_lines = header + extract_qasm(qasm_opt_circuit)
    else:
        qasm_lines = header

    # attach non-clifford gate at the end
    for elem in non_clifford:
        for gate in elem.convert_into_qasm_str():
            qasm_lines.append(f"{gate}")
    with open(filepath, "w") as f:
        f.write("\n".join(qasm_lines))
