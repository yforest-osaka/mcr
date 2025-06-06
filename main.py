from mcr.optimize import (
    fasttodd_optimization,
    pyzx_optimization,
    tket_optimization,
    tmerge_optimization,
)
from mcr.rotation_circuit import PauliRotationSequence
from mcr.unoptimize import unoptimize_circuit


def main():
    ### Input information ###

    num_samples = 1
    nqubits = 4

    with_swap_option = True  # If True, the MCR swap is executed (then the unoptimized circuit becomes longer)
    # Number of iterations for the unoptimized circuit
    unopt_iteration_count = nqubits**2
    ############################

    # Example (4 qubit quantum circuit): [R_{ZZZZ}(+pi/4), R_{ZZZZ}(-pi/4)]
    input_seq = PauliRotationSequence(nqubits)
    initial_pauli_string = "Z" * nqubits
    input_seq.add_gate((0,), f"+{initial_pauli_string}")
    input_seq.get_all()

    # duplicate the circuit
    initial_seq = input_seq.duplicate()

    # Perform unoptimization
    unopt_seq = unoptimize_circuit(input_seq, unopt_iteration_count, with_swap_option)
    if nqubits <= 4:
        assert unopt_seq.is_equivalent(initial_seq), (
            "The circuit is not equivalent to the original one."
        )

    # Save input and output circuits by QASM format
    input_circuit_filepath = "circuit_data/input_circuit.qasm"
    unopted_circuit_filepath = "circuit_data/unopted_circuit.qasm"

    initial_seq.save_qasm(input_circuit_filepath)
    unopt_seq.save_qasm(unopted_circuit_filepath)

    # Compiler evaluation
    # Pytket optimization
    df_tket = tket_optimization(input_circuit_filepath, unopted_circuit_filepath)

    # PyZX optimization
    df_pyzx = pyzx_optimization(input_circuit_filepath, unopted_circuit_filepath)

    # TMerge optimization (Note that arguments are PauliRotationSequence)
    df_tmerge = tmerge_optimization(
        nqubits=nqubits, input_seq=initial_seq, unopted_seq=unopt_seq
    )

    # FastTODD optimization (long execution time)
    df_fasttodd = fasttodd_optimization(unopted_circuit_filepath)

    print("=== Pytket optimization ===")
    print(df_tket)
    print("=== PyZX optimization ===")
    print(df_pyzx)
    print("=== TMerge optimization ===")
    print(df_tmerge)
    print("=== FastTODD optimization ===")
    print(df_fasttodd)

    print("Finished the optimization of the circuit")


if __name__ == "__main__":
    main()
