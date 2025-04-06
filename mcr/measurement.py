import numpy as np

# def gen_proj_matrix(circuit_qubit_count: int, meas_qubit_index_list: list) -> np.ndarray:
#     """
#     Function to generate a projection matrix after partially measuring qubits.

#     Args:
#         circuit_qubit_count (int): Number of qubits in the circuit.
#         meas_qubit_index_list (list): List of indices of qubits to be measured.

#     Returns:
#         np.ndarray: The projection matrix.
#     """
#     ket0 = np.array([[1], [0]])
#     if circuit_qubit_count - 1 in meas_qubit_index_list:
#         mat = ket0
#     else:
#         mat = np.eye(2)
#     for i in reversed(range(0, circuit_qubit_count - 1)):
#         if i in meas_qubit_index_list:
#             mat = np.kron(mat, ket0)
#         else:
#             mat = np.kron(mat, np.eye(2))
#     return mat


def gen_projection_matrix(circuit_qubit_count: int, meas_qubit_info: dict, input_or_output="in") -> np.ndarray:
    state_dict = {
        "0": np.array([[1], [0]]),
        "1": np.array([[0], [1]]),
        "+": np.array([[1], [1]]) / np.sqrt(2),
        "-": np.array([[1], [-1]]) / np.sqrt(2),
    }
    meas_qubit_index_list = list(meas_qubit_info.keys())

    if circuit_qubit_count - 1 in meas_qubit_index_list:
        mat = state_dict[meas_qubit_info[circuit_qubit_count - 1]]
    else:
        mat = np.eye(2)
    for i in reversed(range(0, circuit_qubit_count - 1)):
        if i in meas_qubit_index_list:
            mat = np.kron(mat, state_dict[meas_qubit_info[i]])
        else:
            mat = np.kron(mat, np.eye(2))
    if input_or_output == "in":
        return mat
    elif input_or_output == "out":
        return mat.T
    else:
        raise ValueError('input_or_output must be either "in" or "out"')
