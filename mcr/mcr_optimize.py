from itertools import combinations
import numpy as np
from mcr.gate_apply import PauliBit, multiply_all


# D=ABCとなるパターンを調べる→符号を確認してCliffordの追加が必要かどうかを判断する
# っていうような形に書き換えたい
def find_mcr(
    left_bits: list[PauliBit],
    right_bits: list[PauliBit],
) -> list[tuple[str, str, str, str]]:
    length_left = len(left_bits)
    length_right = len(right_bits)
    for i in range(length_left - 1):
        # print(i)
        for j in range(i + 1, length_left):
            pauli_A = left_bits[i]
            pauli_B = left_bits[j]
            # print(pauli_A.get_pauli_str(), pauli_B.get_pauli_str())

            for k in range(length_right - 1):
                pauli_C = right_bits[k]
                # (1) C は A, B とそれぞれ反可換
                if any([pauli_A.commutes(pauli_C), pauli_B.commutes(pauli_C)]):
                    continue

                coef_abc, pat_abc = multiply_all([pauli_A, pauli_B, pauli_C])
                # ±i が残る場合は除外
                if abs(coef_abc.imag) > 1e-12:
                    continue
                if coef_abc == 1:
                    sgn = 0
                else:
                    assert coef_abc == -1
                    sgn = 1

                # (3) 積の文字列ABC が right_bits に存在するか？
                for l in range(k + 1, length_right):
                    # D = ABC の形になるか？
                    target = right_bits[l]
                    target_sign = target.sgn
                    target_str = target.get_pauli_str()
                    if target_str == pat_abc:  # mcrを発見！
                        if abs(pauli_A.get_angle()) != abs(pauli_B.get_angle()) or abs(
                            pauli_C.get_angle()
                        ) != abs(target.get_angle()):
                            continue

                        if target_sign == sgn:  # D = ABC
                            #! ここをupdate(Aを分割する)
                            # print("update_MCR")
                            # print(
                            #     f"left_bits: {left_bits[i], left_bits[j]}, right_bits: {right_bits[k], right_bits[l]}"
                            # )
                            new_left = left_bits.copy()
                            new_right = right_bits.copy()
                            new_left.pop(j)
                            new_left.pop(i)
                            angle = pauli_A.get_angle()
                            pat_A = pauli_A.get_pauli_str()
                            additional_clifford = PauliBit(pat_A, 2 * angle)
                            new_left += [additional_clifford, pauli_C, target]
                            new_right.pop(l)
                            new_right.pop(k)

                            return (
                                new_left,
                                [
                                    PauliBit(pat_A, -1 * angle),
                                    pauli_B,
                                ]
                                + new_right,
                            )
                        else:
                            assert target_sign + sgn == 1, (
                                "Sign mismatch"
                            )  # D = -ABC (only swap)
                            # print("only swap")
                            # print(
                            #     f"left_bits: {left_bits[i], left_bits[j]}, right_bits: {right_bits[k], right_bits[l]}"
                            # )
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
    left_bits: list[PauliBit], center_bits: list[PauliBit], right_bits: list[PauliBit]
):
    # ) -> tuple(list[PauliBit], list[PauliBit], list[PauliBit]):
    # check  A | B, C | D = D | B, C | A
    # とりあえず中央の要素数は2に限定
    if len(center_bits) != 2:
        return None
    pauli_B, pauli_C = center_bits
    angle_b = pauli_B.get_angle()
    angle_c = pauli_C.get_angle()
    if abs(angle_b) != abs(angle_c):
        return None
    solutions = []
    for l_idx, pauli_A in enumerate(left_bits):
        angle_a = pauli_A.get_angle()
        parity_lst = [
            pauli_A.commutes(pb) for pb in center_bits
        ]  # {A, B}, {A, C}をチェック
        if any(parity_lst):  # 1つでもcommuteするものがいる場合はskip
            continue
        # calc D = ABC and check D in right_bits
        coef_abc, pat_abc = multiply_all([pauli_A, pauli_B, pauli_C])

        # ±i が残る場合は除外
        if abs(coef_abc.imag) > 1e-12:
            continue

        if coef_abc == 1:
            sgn = 0
        else:
            assert coef_abc == -1
            sgn = 1

        # phase_d, pat_dがright_bitsにいるか？

        for r_idx, pauli_D in enumerate(right_bits):
            target_str = pauli_D.get_pauli_str()
            angle_d = pauli_D.get_angle()
            if abs(angle_d) != abs(angle_a):
                continue
            if target_str == pat_abc:  # 発見！
                angle = pauli_D.get_angle()
                sign_d = pauli_D.sgn
                if sgn == sign_d:  # 符号も一致する場合 (D = ABC, only swap)
                    # そのまま場所を入れ替えても等価
                    # print("3layer swap only")
                    # print(
                    #     f"left_bits: {left_bits[l_idx]}, right_bits: {right_bits[r_idx]}"
                    # )
                    left_bits.pop(l_idx)
                    right_bits.pop(r_idx)
                    # return (
                    #     left_bits + [pauli_D],
                    #     center_bits,
                    #     [pauli_A] + right_bits,
                    # )
                    solutions.append(
                        [left_bits + [pauli_D], center_bits, [pauli_A] + right_bits]
                    )
                else:
                    assert sgn + sign_d == 1, "Sign mismatch"  # 符号は逆 (D = -ABC)
                    # print(
                    #     f"left_bits: {left_bits[l_idx]}, right_bits: {right_bits[r_idx]}"
                    # )
                    left_bits.pop(l_idx)
                    right_bits.pop(r_idx)
                    #! ここをupdate(Aを分割する)
                    # print("update_3layer swap")
                    angle_a = pauli_A.get_angle()
                    pat_A = pauli_A.get_pauli_str()
                    new_left = left_bits + [
                        PauliBit(pat_A, 2 * angle_a),
                        pauli_D,
                    ]
                    new_right = [PauliBit(pat_A, -1 * angle_a)] + right_bits
                    solutions.append([new_left, center_bits, new_right])
                    # return new_left, center_bits, new_right
    if solutions:
        # if len(solutions):
        #     print(solutions)
        return solutions[0]
    return None
