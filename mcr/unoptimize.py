import random

import numpy as np
import stim

from mcr.circuit_ops import assign_gate_id_of_generated_gates


def generate_random_pauli_string(num_qubits: int) -> stim.PauliString:
    """Function to generate a random PauliString for the specified number of qubits.

    Args:
        num_qubits (int): Number of qubits

    Returns:
        stim.PauliString: Randomly generated PauliString
    """
    assert num_qubits >= 2, "num_qubits must be greater than or equal to 2."
    # Candidates for Pauli operators
    paulis = ["I", "X", "Y", "Z"]

    flag = True
    while flag:
        # Randomly select Pauli operators
        random_paulis = [random.choice(paulis) for _ in range(num_qubits)]
        join_paulis = "".join(random_paulis)
        if join_paulis not in ["I" * num_qubits]:
            flag = False
    sgn = random.choice(["+", "-"])
    # Generate PauliString
    return stim.PauliString(sgn + join_paulis)


def gen_nontrivial_identity(nqubits):
    """Generate a set of Pauli operators that form a non-trivial identity."""
    pauli_a = generate_random_pauli_string(nqubits)
    flag_b = True
    while flag_b:
        pauli_b = generate_random_pauli_string(nqubits)
        if pauli_a[1:] != pauli_b[1:] and pauli_a.commutes(pauli_b):
            flag_b = False
    flag_c = True
    while flag_c:
        pauli_c = generate_random_pauli_string(nqubits)
        if not pauli_a.commutes(pauli_c) and not pauli_b.commutes(pauli_c):
            flag_c = False
    pauli_d = -1 * pauli_a * pauli_b * pauli_c
    return [
        pauli_a,
        pauli_b,
        pauli_c,
        pauli_d,
        -1 * pauli_a,
        -1 * pauli_b,
        -1 * pauli_c,
        -1 * pauli_d,
    ]


def is_edge(index, gate_id_list):
    """
    Check if the gate at the given index is at the edge of the circuit.
    """
    return index == len(gate_id_list) - 1


def gen_nontrivial_identity_from_gate_info(
    nqubits, left_gate, right_gate=None, with_swap_option=True
):
    """
    Generate a non-trivial identity gate and optionally include Swap operations.
    Retries up to max_trial times if conditions are not met.
    """
    max_trial = 1000
    for _ in range(max_trial):
        gates = gen_nontrivial_identity(nqubits)
        commutes_conditions = [
            left_gate.commutes(gates[0]),
            left_gate.commutes(gates[1]),
        ]
        if right_gate:
            commutes_conditions += [
                right_gate.commutes(gates[-1]),
                right_gate.commutes(gates[-2]),
            ]
        if not any(commutes_conditions):
            break
    else:
        return "Nothing"

    if not with_swap_option:
        return [left_gate] + gates + ([right_gate] if right_gate else [])

    left_additional_gate = -1 * gates[0] * gates[1] * left_gate
    if right_gate:
        right_additional_gate = -1 * gates[-2] * gates[-1] * right_gate
        result_gates = (
            [-1 * left_additional_gate]
            + gates[:2]
            + [left_additional_gate, left_gate]
            + gates[2:6]
            + [right_gate, right_additional_gate]
            + gates[6:]
            + [-1 * right_additional_gate]
        )
    else:
        result_gates = (
            [-1 * left_additional_gate]
            + gates[:2]
            + [left_additional_gate, left_gate]
            + gates[2:]
        )
    return result_gates


def process_gate_replacement(
    circuit, base_id, base_gate, gate_ids, nqubits, with_swap_option
):
    """
    Handles the deletion of the target gate and adds the generated gates to the circuit.
    """
    try:
        left_position = gate_ids.index(base_id)
        left_gate_id = gate_ids[left_position]
        is_at_edge = is_edge(left_position, gate_ids)

        if not is_at_edge:  # Not an edge gate
            right_gate_id = gate_ids[left_position + 1]
            right_gate = circuit.get_gate_from_gate_id(right_gate_id)
            target_gates = gen_nontrivial_identity_from_gate_info(
                nqubits,
                left_gate=base_gate,
                right_gate=right_gate,
                with_swap_option=with_swap_option,
            )
        else:  # Edge gate
            target_gates = gen_nontrivial_identity_from_gate_info(
                nqubits, left_gate=base_gate, with_swap_option=with_swap_option
            )

        if target_gates == "Nothing":
            raise ValueError("Failed to generate nontrivial identity within max trials")

        circuit.delete_from_gate_id(left_gate_id)
        if not is_at_edge:
            circuit.delete_from_gate_id(right_gate_id)

        gates_with_ids = assign_gate_id_of_generated_gates(target_gates, base_id)
        for gate_id, gate in gates_with_ids:
            circuit.add_gate(gate_id, gate)
        return circuit

    except Exception as e:
        return "Nothing"


def unoptimize_circuit(circuit, iteration, with_swap_option):
    """
    Perform rotational unoptimization on the circuit for a given number of iterations.
    """
    nqubits = circuit.get_qubit_count()
    for _ in range(iteration):
        # Randomly select a gate
        while True:
            seed = np.random.randint(0, len(circuit))
            if seed not in circuit.del_nums:
                break
        base_id, base_gate = circuit[seed]
        gate_ids = sorted(circuit.get_all_ids())
        circuit = process_gate_replacement(
            circuit, base_id, base_gate, gate_ids, nqubits, with_swap_option
        )
        if isinstance(circuit, str):
            return "Nothing"
    return circuit
