from itertools import combinations
import numpy as np
from mcr.gate_apply import PauliBit

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


def _build(sign: int, pattern: str) -> str:
    """(±1, 'XYZ…') → '+/-XYZ…' へ戻す（+ も付ける）"""
    return ("-" if sign < 0 else "+") + pattern


def _mult(op1, op2):
    """(±1,'P'),(±1,'Q') → (phase, pattern)"""
    (s1, p1), (s2, p2) = op1, op2
    phase, out, extra = s1 * s2, [], 1
    for a, b in zip(p1, p2):
        c, phi = _TABLE[(a, b)]
        out.append(c)
        extra *= phi
    return phase * extra, "".join(out)


def find_mcr(
    left_bits: list[PauliBit],
    right_bits: list[PauliBit],
) -> list[tuple[str, str, str, str]]:
    """
    left_bits, right_bits は PauliBit インスタンスのリスト。
    元の find_quadruples と同じ (A, B, C, D) のタプルリストを返します。
    出力文字列はすべて get_pauli_str(with_sgn=True) 形式です。
    """

    # —— ヘルパー ——
    def _sign(pb: PauliBit) -> int:
        return 1 if pb.get_angle() > 0 else -1

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
