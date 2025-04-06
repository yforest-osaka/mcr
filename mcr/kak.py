import numpy as np
from qulacs.gate import DenseMatrix, ParametricPauliRotation  # type: ignore
from scipy.linalg import svd

# KAK定数行列の定義
KAK_MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex) * np.sqrt(0.5)

KAK_MAGIC_DAG = np.array([[1, 0, 0, 1], [0, -1j, -1j, 0], [0, 1, -1, 0], [-1j, 0, 0, 1j]], dtype=complex) * np.sqrt(0.5)

KAK_GAMMA = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [-1, 1, -1, 1], [1, -1, -1, 1]], dtype=complex) * 0.25


def kron_factor_4x4_to_2x2s(matrix: np.ndarray) -> tuple:
    """4x4行列U = kron(A, B)をA、B、およびグローバル係数に分割します。

    行列は2x2のユニタリ行列のクロネッカー積である必要があります。
    行列の行列式はゼロでない必要があります。
    不正な行列を与えると、不正な出力が生成されます。

    Args:
        matrix: 分解する4x4ユニタリ行列。

    Returns:
        スカラー係数と2x2行列のペア。3つの行列のクロネッカー積は与えられた行列と等しいです。

    Raises:
        ValueError:
            指定された行列を2x2のピースにテンソル分解できません。
    """

    # Use the entry with the largest magnitude as a reference point.
    a, b = max(((i, j) for i in range(4) for j in range(4)), key=lambda t: abs(matrix[t]))

    # Extract sub-factors touching the reference cell.
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = matrix[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = matrix[a ^ i, b ^ j]

    # Rescale factors to have unit determinants.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 /= np.sqrt(np.linalg.det(f1)) or 1
        f2 /= np.sqrt(np.linalg.det(f2)) or 1

    # Determine global phase.
    g = matrix[a, b] / (f1[a >> 1, b >> 1] * f2[a & 1, b & 1])
    if np.real(g) < 0:
        f1 *= -1
        g = -g
    return g, f1, f2


def bidiagonalize_unitary_with_special_orthogonals(
    mat: np.ndarray,
) -> tuple:  # left,rightの行列式が1になるように補正も同時に行っている
    """特異値分解を行う関数

    Args:
        mat (np.ndarray): 特異値分解を行いたい行列

    Returns:
        tuple: 特異値分解された行列
    """
    matA = mat.real
    left, _, right = svd(matA)  # 実部だけを特異値分解
    # 行列式の調整
    sgn = 1
    if np.linalg.det(left).real < 0:
        left[:, 0] *= -1
        # print('chenged_left!')
        # diag[0, 0] *= -1
        sgn *= -1
    if np.linalg.det(right).real < 0:
        right[0, :] *= -1
        # diag[0, 0] *= -1
        # print('chenged_right!')
        sgn *= -1
    # 対角成分を抽出
    diag = left.T @ mat @ right.T
    # print('diag: ',diag)
    # print('mat: ',mat)
    diag *= sgn
    if sgn == -1:
        left *= -1
    # print('diag_aft: ',diag)
    # print('svd: ',left@diag@right) #どうやらsgnが-1の時に値が一致しなくなるらしい
    return left, diag, right


def is_so4_matrix(matrix: np.ndarray) -> bool:  # SO(4)かどうかを判定する関数
    """SO(4)行列かどうかを判定する関数

    Args:
        matrix (np.ndarray): 判定したい行列

    Returns:
        bool: SO(4)行列かどうか
    """
    # Check if the matrix is 4x4
    if matrix.shape != (4, 4):
        return False

    # Check if the matrix is orthogonal: A^T A = I
    if not np.allclose(np.dot(matrix.T, matrix), np.identity(4)):
        return False

    # Check if the determinant is 1
    if not np.isclose(np.linalg.det(matrix), 1):
        return False

    return True


def close(matA: np.ndarray, matB: np.ndarray) -> bool:
    """行列が近いかどうかを判定する関数

    Args:
        matA (np.ndarray): 行列A
        matB (np.ndarray): 行列B

    Returns:
        bool: 行列が近いかどうか
    """
    return np.allclose(np.round(matA, 5), np.round(matB, 5))


def kak_decomposition(mat: np.ndarray, target_index: list) -> list:
    """KAK分解を行う関数

    Args:
        mat (np.ndarray): 行列
        target_index (list): ターゲットのインデックス

    Returns:
        list: KAK分解された行列の入ったリスト
    """
    result = []
    o_l, d, o_r = bidiagonalize_unitary_with_special_orthogonals(KAK_MAGIC_DAG @ mat @ KAK_MAGIC)  # 特異値分解
    assert is_so4_matrix(o_l)
    assert is_so4_matrix(o_r)
    _, a0, a1 = kron_factor_4x4_to_2x2s(KAK_MAGIC @ o_l @ KAK_MAGIC_DAG)
    _, b0, b1 = kron_factor_4x4_to_2x2s(KAK_MAGIC @ o_r @ KAK_MAGIC_DAG)
    # 対角成分の処理
    D = np.diag(d)
    phase_angle = np.angle(D)
    A = np.array([[1, 1, -1, 1], [1, 1, 1, -1], [1, -1, -1, -1], [1, -1, 1, 1]])
    B = phase_angle
    # 連立方程式を解く
    solution = np.linalg.solve(A, B)
    _, x, y, z = solution.real
    angle_x, angle_y, angle_z = 2 * x, 2 * y, 2 * z

    upper = target_index[0]
    lower = target_index[1]
    result.append([(0, 0), DenseMatrix(lower, b0)])
    result.append([(0, 1), DenseMatrix(upper, b1)])
    result.append([(1, 0), ParametricPauliRotation([upper, lower], [3, 3], angle_z)])
    result.append([(1, 1), ParametricPauliRotation([upper, lower], [2, 2], angle_y)])
    result.append([(1, 2), ParametricPauliRotation([upper, lower], [1, 1], angle_x)])
    result.append([(2, 0), DenseMatrix(lower, a0)])
    result.append([(2, 1), DenseMatrix(upper, a1)])
    return result
