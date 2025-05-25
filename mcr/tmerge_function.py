import stim
from tqdm import tqdm


def apply_pauli_gates(circuit, qubit_indices, pauli_ids, right_side=False):
    """Apply Pauli gates to the specified qubits."""
    # print(qubit_indices, pauli_ids)
    pauli_ids = pauli_ids[
        1:
    ]  # Exclude the first character as it contains sign information
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


def conversion_from_pauli_to_circuit(stim_pauli):
    sgn = stim_pauli.sign
    # Get the indices of qubits containing Pauli operators
    non_identity_pauli_indices = stim_pauli.pauli_indices()
    position = max(non_identity_pauli_indices)  # The last non-zero index
    circuit = stim.Circuit()

    apply_pauli_gates(
        circuit, non_identity_pauli_indices, str(stim_pauli), right_side=False
    )
    if len(non_identity_pauli_indices) >= 2:  # CNOT is required
        # Apply CNOT gates
        for idx in non_identity_pauli_indices:
            if idx != position:
                circuit.append("CNOT", [idx, position])

    # Apply rotation gates
    if sgn.real > 0:
        circuit.append("S", [position])
    else:
        circuit.append("S_DAG", [position])

    if len(non_identity_pauli_indices) >= 2:  # CNOT is required
        # Apply CNOT gates in reverse order
        for idx in reversed(non_identity_pauli_indices):
            if idx != position:
                circuit.append("CNOT", [idx, position])

    # Apply Pauli gates in reverse order
    apply_pauli_gates(
        circuit, non_identity_pauli_indices, str(stim_pauli), right_side=True
    )
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


# Update the Pauli string based on the given Clifford circuit
def clifford_update(clifford_circuit, stim_pauli_str):
    new_val = stim_pauli_str.after(clifford_circuit)
    return new_val


def optimization_process(
    target_str: stim.PauliString, data_all: list[stim.PauliString]
):
    for i, ele in enumerate(reversed(data_all)):
        relation = get_rotation_relation(target_str, ele)
        if relation == "commute":
            continue
        elif relation == "anti-commute":
            return "Nothing"
        elif relation == "zero":
            data_all = [
                data_all[j] for j in range(len(data_all)) if j != len(data_all) - i - 1
            ]
            # print(len(data_all))
            return data_all
        elif relation in {"plus_clifford", "minus_clifford"}:
            # print("Delete_for_clifford!!", len(data_all) - i - 1, ele)
            # print(data_all)
            data_all = [
                data_all[j] for j in range(len(data_all)) if j != len(data_all) - i - 1
            ]
            # print(data_all)
            # print(f"be clifford: {target_str}")
            new_clifford_circuit = conversion_from_pauli_to_circuit(
                target_str
            )  # A mechanism to add the merged PauliRotation is required!
            return data_all, new_clifford_circuit
    return "Nothing"


# Implementation of the algorithm to divide into T layers (arXiv:2407.08695 Algorithm 1)
def stim_grouping_of_pauli_rotations(stim_data_lst, joint=False):
    # length = len(stim_data_lst)
    # print('length', length)
    L = []  # Empty list L
    for Rp in stim_data_lst:
        # print('Rp', Rp)
        j = 0  # Initialize (create a new layer if no anti-commuting group is found)
        for k in reversed(range(len(L))):
            # print('k', k)
            # Check if anti-commuting
            commute_info = [Rp.commutes(Rk) for Rk in L[k]]
            if not all(commute_info):  # If there is even one anti-commuting element
                j = k + 1
                # print(f"Not all commute: {j}")
                break
            else:
                # print(f'All commute! {k}')
                pass
        if j == 0:
            # Create a new group
            if len(L) == 0:
                L.append([Rp])
            else:
                L[0].append(Rp)
        else:
            # print(f"j is not -1: {j}")
            if len(L) == j:
                # Create a new layer
                L.append([Rp])
            else:
                # Add to an existing group
                L[j].append(Rp)
    if joint:
        return [
            item for sublist in L for item in sublist
        ]  # Flatten all elements of L and return
    else:
        return L


def zhang_optimization(stim_data_lst):
    clifford_circuit = stim.Circuit()
    optimized_rotations = []
    for ele in tqdm(reversed(stim_data_lst), desc="Zhang optimization", leave=False):
        # If the Clifford circuit has been updated, apply it to the current element
        if len(clifford_circuit) > 0:
            ele = clifford_update(clifford_circuit, ele)

        # If there are no optimized rotations yet, add the current element
        if len(optimized_rotations) == 0:
            optimized_rotations.append(ele)
        else:
            # Check the relationship with already appended Pauli rotations
            value = optimization_process(ele, optimized_rotations)
            if len(value) == 2 and isinstance(value, tuple):
                # If a new Clifford circuit (+pi/2 or -pi/2) is generated
                optimized_rotations, new_clifford_circuit = value
                clifford_circuit = clifford_circuit + new_clifford_circuit
            else:
                if value == "Nothing":
                    optimized_rotations.append(ele)
                else:
                    # If the angles cancel each other out to 0
                    optimized_rotations = value

    # Reverse the optimized rotations at the end
    optimized_rotations = optimized_rotations[::-1]
    return optimized_rotations, clifford_circuit


def zhang_optimization_until_convergence(
    nqubits, stim_data_lst, with_grouping_t_layers=False, with_process=False
):
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
            # Combine all circuits in cliff_circs & update the circuit
            combined_clifford_circuit = stim.Circuit()
            for i, circuit in enumerate(cliff_circs):
                combined_clifford_circuit += circuit
                for j in range(i):
                    target_non_cliffords = opt_rots_lst[j]
                    opt_rots_lst[j] = [
                        ele.after(circuit) for ele in target_non_cliffords
                    ]
            clifford_circuit = combined_clifford_circuit
            optimized_rotations = [
                item for sublist in opt_rots_lst for item in sublist
            ]  # Flatten the list again here
        #####
        else:
            optimized_rotations, clifford_circuit = zhang_optimization(stim_data_lst)

        if len(optimized_rotations) >= length:
            flag = False
        if len(clifford_circuit) > 0:
            clifford_data.append(clifford_circuit)
        counter += 1
        stim_data_lst = optimized_rotations.copy()
        # stim_data_lst = stim_grouping_of_pauli_rotations(optimized_rotations, joint=True) # Apply grouping if needed
        # print('Grouping done!')
        length = len(optimized_rotations)
        # Sequentially add clifford_data from left to right
        combined_clifford_circuit = stim.Circuit()
        for circuit in clifford_data:
            combined_clifford_circuit += circuit
        # All circuits are combined into combined_clifford_circuit
    if with_process:
        return optimized_rotations, combined_clifford_circuit, non_clifford_gate_counts
    else:
        return optimized_rotations, combined_clifford_circuit
