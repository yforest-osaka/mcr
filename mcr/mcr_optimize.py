from itertools import combinations
import numpy as np
from mcr.gate_apply import PauliBit
from mcr.equiv_check import pauli_bit_equivalence_check

# ── 1. 1-qubit の積テーブル ──────────────────────────────
_TABLE = {
    ("I", "I"): ("I", 1),
    ("I", "X"): ("X", 1),
    ("I", "Y"): ("Y", 1),
    ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1),
    ("Y", "I"): ("Y", 1),
    ("Z", "I"): ("Z", 1),
    ("X", "X"): ("I", 1),
    ("Y", "Y"): ("I", 1),
    ("Z", "Z"): ("I", 1),
    ("X", "Y"): ("Z", 1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "X"): ("Y", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Z", "Y"): ("X", -1j),
    ("X", "Z"): ("Y", -1j),
}


def _mult(op1, op2):
    """(±1,'P'),(±1,'Q') → (phase, pattern)"""
    (s1, p1), (s2, p2) = op1, op2
    phase, out, extra = s1 * s2, [], 1
    for a, b in zip(p1, p2):
        c, phi = _TABLE[(a, b)]
        out.append(c)
        extra *= phi
    return phase * extra, "".join(out)


# —— ヘルパー ——
def _sign(pb: PauliBit) -> int:
    return 1 if pb.get_angle() > 0 else -1


def find_mcr(
    left_bits: list[PauliBit],
    right_bits: list[PauliBit],
) -> list[tuple[str, str, str, str]]:
    """
    left_bits, right_bits は PauliBit インスタンスのリスト。
    元の find_quadruples と同じ (A, B, C, D) のタプルリストを返します。
    出力文字列はすべて get_pauli_str(with_sgn=True) 形式です。
    """

    # —— 右側パターン→符号集合マップ ——
    right_map: dict[str, set[int]] = {}
    for pb in right_bits:
        pat = pb.get_pauli_str()  # 符号抜きパターン
        right_map.setdefault(pat, set()).add(_sign(pb))

    solutions = []
    seen = set()

    # —— A,B を組み合わせて探索 ——
    for i, j in combinations(range(len(left_bits)), 2):
        A, B = left_bits[i], left_bits[j]
        A_raw = A.get_pauli_str(with_sgn=True)
        B_raw = B.get_pauli_str(with_sgn=True)

        for C in right_bits:
            C_raw = C.get_pauli_str(with_sgn=True)

            # (1) C は A, B とそれぞれ反可換
            if not (not A.commutes(C) and not B.commutes(C)):
                continue

            # (2) D = –ABC を計算
            phase_ab, pat_ab = _mult(
                (_sign(A), A.get_pauli_str()),
                (_sign(B), B.get_pauli_str()),
            )
            phase_abc, pat_abc = _mult(
                (phase_ab, pat_ab),
                (_sign(C), C.get_pauli_str()),
            )
            phase_d = -phase_abc

            # ±i が残る場合は除外
            if abs(phase_d.imag) > 1e-12:
                continue
            sign_d = 1 if phase_d.real > 0 else -1

            # (3) D が right_bits に存在するか？
            if pat_abc not in right_map or sign_d not in right_map[pat_abc]:
                continue

            D_raw = ("+" if sign_d > 0 else "-") + pat_abc
            if D_raw == C_raw:
                continue

            # —— 重複除去 ——
            key = (
                tuple(sorted((A_raw, B_raw))),
                frozenset((C_raw, D_raw)),
            )
            if key in seen:
                continue
            seen.add(key)

            # 出力は C,D を辞書順に
            C_out, D_out = sorted((C_raw, D_raw))
            solutions.append((A_raw, B_raw, C_out, D_out))

    return solutions


def find_nontrivial_swap(
    left_bits: list[PauliBit], center_bits: list[PauliBit], right_bits: list[PauliBit]
):
    # ) -> tuple(list[PauliBit], list[PauliBit], list[PauliBit]):
    # check  A | B, C | D = D | B, C | A
    # とりあえず中央の要素数は2に限定
    if len(center_bits) != 2:
        return None

    for l_idx, left_bit in enumerate(left_bits):
        parity_lst = [
            left_bit.commutes(pb) for pb in center_bits
        ]  # {A, B}, {A, C}をチェック
        if any(parity_lst):  # 1つでもcommuteするものがいる場合はskip
            continue
        # calc D = ABC and check D in right_bits
        A = left_bit.get_pauli_str(with_sgn=False)
        B = center_bits[0].get_pauli_str(with_sgn=False)
        C = center_bits[1].get_pauli_str(with_sgn=False)
        A_raw = left_bit.get_pauli_str(with_sgn=True)
        B_raw = center_bits[0].get_pauli_str(with_sgn=True)
        C_raw = center_bits[1].get_pauli_str(with_sgn=True)

        phase_ab, pat_ab = _mult(
            (_sign(left_bit), A),
            (_sign(center_bits[0]), B),
        )
        phase_abc, pat_abc = _mult(
            (phase_ab, pat_ab),
            (_sign(center_bits[1]), C),
        )
        phase_d = phase_abc  # complex number
        pat_d = pat_abc  # pauli_string(without sign)
        # ±i が残る場合は除外
        if abs(phase_d.imag) > 1e-12:
            continue
        # phase_d, pat_dがright_bitsにいるか？
        for r_idx, right_bit in enumerate(right_bits):
            target_pat = right_bit.get_pauli_str(with_sgn=False)

            if target_pat == pat_d:  # 発見！
                angle = right_bit.get_angle()
                sign = np.sign(angle)
                if sign == phase_d:  # 符号も一致する場合
                    # そのまま場所を入れ替えても等価
                    # print("equiv_check in swap case")
                    return (
                        left_bits[:l_idx] + left_bits[l_idx + 1 :] + [right_bit],
                        center_bits,
                        [left_bit] + right_bits[:r_idx] + right_bits[r_idx + 1 :],
                    )
                else:
                    assert sign == -phase_d, (
                        "Sign mismatch: expected sign to be the negative of phase_d"
                    )  # 符号は逆
                    # right_bitsに新たな要素を追加して入れ替える
                    # new_right = (
                    #     [left_bit] + right_bits[:r_idx] + right_bits[r_idx + 1 :]
                    # )

                    new_left = (
                        left_bits[:l_idx]
                        + left_bits[l_idx + 1 :]
                        + [PauliBit(pat_d, -1 * angle)]
                    )
                    new_right = (
                        [left_bit]
                        + [PauliBit(pat_d, 2 * angle)]  # あえて作るClifford
                        + right_bits[:r_idx]
                        + right_bits[r_idx + 1 :]
                    )
                    return new_left, center_bits, new_right
    return None
