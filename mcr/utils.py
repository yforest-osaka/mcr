import math
from collections import Counter
from itertools import permutations, product

import numpy as np
from qulacs.gate import X, Y, Z
from tqdm import tqdm

from mcr.filesave import read_pickle

# from unoptimization.rot_class import RotOps

# from unoptimization.rotation_2q_class import Rot2QOps


# 全候補
def gen_candidates(nqubits=2, repeat=2, exclude_identity=True):
    base_data = list(product([0, 1, 2, 3], repeat=nqubits))
    if exclude_identity:
        base_data.remove(((0,) * nqubits))

    if repeat == 1:
        return [ele for ele in base_data]
    return list(
        product(
            base_data,
            repeat=repeat,
        )
    )


def remove_consecutive(candidates):
    return [c for c in candidates if RotOps(c).has_consecutive() == False]  # 更新(1010)


def tuple_to_matrix(tuple_data):
    # number_to_matrix = {0: np.eye(2), 1: X(0).get_matrix(), 2: Y(0).get_matrix(), 3: Z(0).get_matrix()}
    # tmp = []
    # for num in tuple_data:
    #     tmp.append(number_to_matrix[num])
    # return np.kron(tmp[0], tmp[1])
    number_to_matrix = {0: np.eye(2), 1: X(0).get_matrix(), 2: Y(0).get_matrix(), 3: Z(0).get_matrix()}
    tmp = []
    for num in tuple_data:
        tmp.append(number_to_matrix[num])
    mat = np.eye(1)
    for matrix in tmp:
        mat = np.kron(mat, matrix)
    return mat


def commutator(matrix1, matrix2):
    return matrix1 @ matrix2 - matrix2 @ matrix1


def anti_commutator(matrix1, matrix2):
    return matrix1 @ matrix2 + matrix2 @ matrix1


def objective_function(p1, p2, p3, p4, angle_lst, patrial=False):
    s = np.sin(np.pi / 8)
    c = np.cos(np.pi / 8)
    matrix_info = [p1, p2, p3, p4]
    for i, ele in enumerate(angle_lst):
        if ele < 0:  # -np.pi/4の場合は符号を反転
            matrix_info[i] = -1 * matrix_info[i]

    p1, p2, p3, p4 = matrix_info
    term1 = s**2 * (
        p1 @ commutator(p2, p3) @ p4
        + p1 @ p3 @ commutator(p2, p4)
        + commutator(p1, p3) @ p4 @ p2
        + p3 @ commutator(p1, p4) @ p2
    )
    term2 = (
        0.5
        * np.sqrt(2)
        * 1j
        * (
            commutator(p1, p3) @ (p2 + p4)
            + (p1 + p3) @ commutator(p2, p4)
            + p1 @ commutator(p2, p3)
            + commutator(p1, p4) @ p2
            + p3 @ commutator(p1, p4)
            + commutator(p2, p3) @ p4
        )
    )
    term3 = -(c**2) * (commutator(p1, p3) + commutator(p1, p4) + commutator(p2, p3) + commutator(p2, p4))
    if patrial:
        return term1, term2, term3
    return term1 + term2 + term3
    # return term2


def matrix_equality(matrix1, matrix2):
    return np.allclose(matrix1, matrix2)


def gen_commutation_info(tuple_pauli_ids):
    info = []
    n = len(tuple_pauli_ids)
    rot = RotOps(tuple_pauli_ids)  # 更新(1010)
    for i in range(n):
        for j in range(i + 1, n):
            info.append(rot.is_commute(i, j))
    assert len(info) == math.comb(n, 2), "invalid info length"
    return tuple(info)


def matrix_to_pauli_string(matrix):
    candidates = [ele[0] for ele in gen_candidates(repeat=1)]
    coefs = [1, -1, 1j, -1j]
    for candidate in candidates:
        for coef in coefs:
            if np.allclose(matrix, coef * tuple_to_matrix(candidate)):
                return coef, candidate
    raise ValueError(f"Not found: \n{matrix}")


# Hilbert-Schmidt内積の計算からでもcoefは取り出せるはず
def get_coef_from_matrix(matrix, show_log=False):
    matrix_size = matrix.shape[0]
    candidates = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    ]
    coefs = []
    for candidate in candidates:
        coefs.append(np.trace(matrix @ tuple_to_matrix(candidate)) / matrix_size)
    pauli_str_dicts = {
        (0, 0): "II",
        (0, 1): "IX",
        (0, 2): "IY",
        (0, 3): "IZ",
        (1, 0): "XI",
        (1, 1): "XX",
        (1, 2): "XY",
        (1, 3): "XZ",
        (2, 0): "YI",
        (2, 1): "YX",
        (2, 2): "YY",
        (2, 3): "YZ",
        (3, 0): "ZI",
        (3, 1): "ZX",
        (3, 2): "ZY",
        (3, 3): "ZZ",
    }
    string = ""
    for i, coef in enumerate(coefs):
        if not np.allclose(coef, 0):
            string += f"({coef:.1f}) * {pauli_str_dicts[candidates[i]]}, "
    if show_log:
        print(string)
    if string == "":
        string = "zero"
    return coefs, string


def commute_check(tuple_id1, tuple_id2):
    sgn = 1
    for i in range(len(tuple_id1)):
        target = [tuple_id1[i], tuple_id2[i]]
        if target[0] != target[1] and 0 not in target:  # anti-commute
            sgn *= -1
    if sgn == 1:
        return True
    return False


def all_commutable_in_one_group(data):
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if not commute_check(data[i], data[j]):
                return False
    return True


def satisfies_litinski_condition(tuple_ids1, tuple_ids2):
    assert all_commutable_in_one_group(tuple_ids1), "invalid data"
    assert all_commutable_in_one_group(tuple_ids2), "invalid data"
    for ele1 in tuple_ids1:
        tf_lst = []
        for ele2 in tuple_ids2:
            tf_lst.append(commute_check(ele1, ele2))
        if all(tf_lst):
            return False
    return True


# def all_anti_commutable_in_mutual_group(data_1, data_2):
#     for i in range(len(data_1)):
#         for j in range(len(data_2)):
#             if commute_check(data_1[i], data_2[j]):
#                 return False
#     return True


# def matrix_sum(matrices):
#     n = matrices[0].shape[0]
#     mat = np.zeros((n, n), dtype=complex)
#     for ele in matrices:
#         mat += ele
#     return mat


# def matrix_mult(matrices):
#     n = matrices[0].shape[0]
#     mat = np.zeros((n, n), dtype=complex)
#     for ele in matrices:
#         mat = mat @ ele
#     return mat


# from functools import reduce


# def get_mult(pauli_ids: tuple):
#     # ((0, 0, 1), (0, 1, 0), (1, 0, 0))
#     result = []
#     pauli_id_to_calc_space = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
#     calc_space_to_pauli_id = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
#     qubit_count = len(pauli_ids[0])
#     for i in range(qubit_count):
#         target_data = [pauli_id_to_calc_space[ele[i]] for ele in pauli_ids]
#         value = tuple(map(lambda x: reduce(lambda a, b: a ^ b, x), zip(*target_data)))
#         result.append(calc_space_to_pauli_id[value])
#     return tuple(result)
