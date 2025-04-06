# Clifford+T回路の代数表現に関する関数
import itertools

import numpy as np
from tqdm import tqdm


def get_minimum_exponent(matrix: np.ndarray) -> tuple:
    """最小のexponentを求める

    Args:
        matrix (np.ndarray): 行列

    Returns:
        tuple: (exponentのリスト, 解のリスト)
    """
    omega = np.exp(1j * np.pi / 4)
    sq_omega = omega**2
    cb_omega = omega**3

    candidates = list(itertools.product([-1, 0, 1], repeat=4))
    values = []
    for candidate in candidates:
        a, b, c, d = candidate
        values.append(cb_omega * a + sq_omega * b + omega * c + d)
    entries_count = matrix.size
    matrix_elements = matrix.reshape(1, entries_count)[0]
    k_list = np.full_like(np.zeros(entries_count), -1)
    solutions = []
    max_exponent = 500
    for j in tqdm(range(max_exponent)):
        # jにはexponentが入っている
        coef = np.sqrt(2) ** (j)
        for m, matrix_entry in enumerate(matrix_elements):
            if k_list[m] != -1:
                continue
            else:
                target_value = matrix_entry * coef
                # valuesの中で一致するものがあるか確認する
                for i, value in enumerate(values):
                    if np.allclose(value, target_value):
                        # print('発見！',j, candidates[i])
                        solutions.append(candidates[i])
                        k_list[m] = j
                        break
        if -1 not in k_list:
            break
    return k_list, solutions
