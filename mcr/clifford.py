import numpy as np
import sympy as sp
from IPython.display import Math, display
from qulacs import QuantumCircuit
from qulacs.gate import SWAP, H, Identity, S, Sdag, T, Tdag, X, Y, Z  # type: ignore

CZ_MATRIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
CNOT_MATRIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
SWAP_CNOT_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
x, y, z = X(0).get_matrix(), Y(0).get_matrix(), Z(0).get_matrix()
h, s, t = H(0).get_matrix(), S(0).get_matrix(), T(0).get_matrix()
sdag, tdag = Sdag(0).get_matrix(), Tdag(0).get_matrix()
id = np.eye(2)
cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
swap = SWAP(0, 1).get_matrix()

paulis = [id, x, y, z]
pauli_names = ["I", "X", "Y", "Z"]


def show(matrix: np.ndarray):
    """Display a matrix in a pretty format."""
    return display(Math(sp.latex(sp.Matrix(np.round(matrix, 3)))))


def get_angle(matrix: np.ndarray) -> np.ndarray:
    """行列の角度を取得する

    Args:
        matrix (np.ndarray): 行列

    Returns:
        np.ndarray: 行列の角度
    """
    return np.angle(matrix) / np.pi


def kronecker_product_recursive(matrices: list) -> np.ndarray:
    """行列のリストからテンソル積を計算する

    Args:
        matrices (list): 行列のリスト

    Returns:
        np.ndarray: テンソル積の結果
    """
    if len(matrices) == 1:
        return matrices[0]
    else:
        return np.kron(matrices[0], kronecker_product_recursive(matrices[1:]))


def generate_pauli_combinations(n: int) -> tuple[list, list]:
    """パウリ行列の組み合わせを生成する

    Args:
        n (int): パウリ行列の数

    Returns:
        tuple[list, list]: パウリ行列のリストと名前のリスト
    """
    if n == 1:
        return paulis, pauli_names

    previous_matrices, previous_names = generate_pauli_combinations(n - 1)

    new_matrices = []
    new_names = []

    for i, p in enumerate(paulis):
        for _, (prev_matrix, prev_name) in enumerate(zip(previous_matrices, previous_names)):
            new_matrices.append(np.kron(p, prev_matrix))
            new_names.append(pauli_names[i] + "⊗" + prev_name)

    return new_matrices, new_names


def is_pauli_matrix(matrix: np.ndarray, product_paulis: list) -> bool:
    """パウリ群に含まれるかどうかを判定する

    Args:
        matrix (np.ndarray): 行列
        product_paulis (list): パウリ行列のリスト

    Returns:
        bool: パウリ群に含まれるかどうか
    """
    for pauli in product_paulis:
        # Check if the matrix is a scalar multiple of a Pauli matrix
        if np.allclose(matrix, pauli) or np.allclose(matrix, -pauli):
            return True

        # Check if the matrix is a scalar multiple of a Pauli matrix with a complex phase factor
        for phase in [1j, -1j]:
            if np.allclose(matrix, phase * pauli) or np.allclose(matrix, -phase * pauli):
                return True
    return False


def is_clifford(matrix: np.ndarray) -> bool:
    """Clifford行列かどうかを判定する

    Args:
        matrix (np.ndarray): 行列

    Returns:
        bool: Clifford行列かどうか
    """
    qubit_count = int(np.log2(matrix.shape[0]))
    pauli_products, _ = generate_pauli_combinations(qubit_count)
    for pauli_product in pauli_products:
        mat = matrix @ pauli_product @ matrix.conj().T
        if not is_pauli_matrix(mat, pauli_products):
            return False
    return True


def complement_identity(circuit: QuantumCircuit) -> QuantumCircuit:
    """一個もゲートがないqubit_indexが存在する場合飲みIdentityゲートを追加する

    Args:
        circuit (QuantumCircuit): Qulacsの回路

    Returns:
        QuantumCircuit: Identityゲートを追加した回路
    """
    indices = []
    for i in range(circuit.get_gate_count()):
        indices += circuit.get_gate(i).get_target_index_list()
        indices += circuit.get_gate(i).get_control_index_list()
    # if 0 not in indices:
    #     circuit.add_gate(Identity(0))
    # elif 1 not in indices:
    #     circuit.add_gate(Identity(1))
    for num in range(circuit.get_qubit_count()):
        if num not in indices:
            circuit.add_gate(Identity(num))
    return circuit


def ctrl_gates_to_dem(circuit: QuantumCircuit) -> QuantumCircuit:
    """制御ゲートをDeMに変換する

    Args:
        circuit (QuantumCircuit): Qulacsの回路

    Returns:
        QuantumCircuit: DeMに変換した回路
    """
    GATE_MATRIX_MAP = {"CZ": CZ_MATRIX, "CNOT": CNOT_MATRIX, "SWAP_CNOT": SWAP_CNOT_MATRIX}
    n = circuit.get_qubit_count()
    Circuit_out = QuantumCircuit(n)
    for g in range(circuit.get_gate_count()):
        gate = circuit.get_gate(g)
        gate_name = gate.get_name()
        if gate_name in GATE_MATRIX_MAP:  # CZ or CNOT
            ctrl, tg = gate.get_control_index_list()[0], gate.get_target_index_list()[0]
            if ctrl < tg:
                gate_name = "SWAP_CNOT"
                Circuit_out.add_dense_matrix_gate([ctrl, tg], GATE_MATRIX_MAP[gate_name])
            else:
                Circuit_out.add_dense_matrix_gate([tg, ctrl], GATE_MATRIX_MAP[gate_name])
        else:
            Circuit_out.add_gate(gate)
    return Circuit_out


# deprecated

# def find_scalar(A, B):
#     # Aの非ゼロ要素に対応するBの要素の比を計算
#     ratios = B[A != 0] / A[A != 0]
#     # 比の値がすべて同じであることを確認
#     if np.allclose(ratios, ratios[0]):
#         if abs(ratios[0]) < 0.001:
#             raise ValueError("Answered zero.")
#         else:
#             return ratios[0]
#     else:
#         raise ValueError("A and B are not related by a scalar multiplication.")
