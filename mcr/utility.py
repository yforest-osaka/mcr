# 新しく追加(適宜適切な場所に移動してください)
from qulacs.gate import T, Tdag, H, S, Sdag, CNOT  # type: ignore
from qulacs_core import ClsOneQubitGate
from collections import Counter
from qulacs import QuantumCircuit

# 可換になるゲートの組み合わせ
COMMUTE_DICTS = {
    (0, 1): [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
    (0, 2): [(1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2)],
    (0, 3): [(1, 0), (1, 3), (2, 0), (2, 3), (3, 0), (3, 3)],
    (1, 0): [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
    (1, 1): [(0, 1), (1, 0), (2, 2), (2, 3), (3, 2), (3, 3)],
    (1, 2): [(0, 2), (1, 0), (2, 1), (2, 3), (3, 1), (3, 3)],
    (1, 3): [(0, 3), (1, 0), (2, 1), (2, 2), (3, 1), (3, 2)],
    (2, 0): [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3)],
    (2, 1): [(0, 1), (1, 2), (1, 3), (2, 0), (3, 2), (3, 3)],
    (2, 2): [(0, 2), (1, 1), (1, 3), (2, 0), (3, 1), (3, 3)],
    (2, 3): [(0, 3), (1, 1), (1, 2), (2, 0), (3, 1), (3, 2)],
    (3, 0): [(0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (3, 3)],
    (3, 1): [(0, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 0)],
    (3, 2): [(0, 2), (1, 1), (1, 3), (2, 1), (2, 3), (3, 0)],
    (3, 3): [(0, 3), (1, 1), (1, 2), (2, 1), (2, 2), (3, 0)],
}

ANTI_COMMUTE_DICTS = {
    (0, 1): [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)],
    (0, 2): [(0, 1), (0, 3), (1, 1), (1, 3), (2, 1), (2, 3), (3, 1), (3, 3)],
    (0, 3): [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)],
    (1, 0): [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)],
    (1, 1): [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)],
    (1, 2): [(0, 1), (0, 3), (1, 1), (1, 3), (2, 0), (2, 2), (3, 0), (3, 2)],
    (1, 3): [(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (3, 0), (3, 3)],
    (2, 0): [(1, 0), (1, 1), (1, 2), (1, 3), (3, 0), (3, 1), (3, 2), (3, 3)],
    (2, 1): [(0, 2), (0, 3), (1, 0), (1, 1), (2, 2), (2, 3), (3, 0), (3, 1)],
    (2, 2): [(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2)],
    (2, 3): [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2), (3, 0), (3, 3)],
    (3, 0): [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)],
    (3, 1): [(0, 2), (0, 3), (1, 0), (1, 1), (2, 0), (2, 1), (3, 2), (3, 3)],
    (3, 2): [(0, 1), (0, 3), (1, 0), (1, 2), (2, 0), (2, 2), (3, 1), (3, 3)],
    (3, 3): [(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)],
}


# インポート
def add_T_orTdag(seed:int=1, index:int) -> ClsOneQubitGate:
    """TゲートもしくはTdagゲートを返す

    Args:
        index (int): qubit_index
        seed (int, optional): Tなら1、Tdagなら-1(1以外)を指定する. Defaults to 1.

    Returns:
        ClsOneQubitGate: T or Tdag gate
    """
    if seed == 1:
        return T(index)
    else:
        return Tdag(index)


def check_commute(string:str)->list:
    """文字列から可換性を判定する

    Args:
        string (str): ゲートの情報が書かれた文字列

    Returns:
        list: 可換ならTrue, 反可換ならFalseが入ったリスト
    """
    result = []
    for i in range(len(string) - 1):
        upper, lower = [string[i][0][0], string[i + 1][0][0]], [string[i][0][1], string[i + 1][0][1]]
        sgn = 1
        for ele in [upper, lower]:
            if ele[0] != ele[1] and 0 not in ele:  # anti-commute
                sgn *= -1
        if sgn == 1:
            result.append(True)
        else:
            result.append(False)
    return result


def has_consecutive_duplicates(lst:list)->bool:
    """連続して同じ要素が並んでいる箇所が存在するかどうかを判定する

    Args:
        lst (list): リスト

    Returns:
        bool: 連続して同じ要素が並んでいる箇所が存在するかどうか
    """
    for i in range(len(lst) - 1):
        if lst[i] == lst[i + 1]:
            return True
    return False


def has_duplicates(lst:list)->bool:
    """リスト内に重複する要素が存在するかどうかを判定する

    Args:
        lst (list): リスト

    Returns:
        bool: 重複する要素が存在するかどうか
    """
    if len(Counter([str(ele) for ele in lst])) < len(lst):
        return True
    else:
        return False


def convert_to_clifford_t_circuit(string_list:list)->QuantumCircuit: # 等価な回路を実現するにはゲート列を逆にしないといけない気がする
    """(paui_id, angle)のリストを受け取り、Clifford + T回路に変換する

    Args:
        string_list (list): (pauli_id, angle)のリスト

    Returns:
        QuantumCircuit: Clifford + T回路
    """
    circuit = QuantumCircuit(2)
    # pauli_idはそのまま&回転角を反転させる
    for gate in string_list:
        # pauli_ids = list(reversed(gate[0]))
        pauli_ids = gate[0]
        angle = -1 * gate[1]
        t_id = 0
        if angle > 0:  # t gate
            t_id = 1
        # pauli_idに0が入っているか否かで場合わけ
        if 0 not in pauli_ids:  # 要CNOT
            for idx, pauli_id in zip([0, 1], pauli_ids):
                if pauli_id == 1:
                    circuit.add_gate(H(idx))
                elif pauli_id == 2:
                    circuit.add_gate(Sdag(idx))
                    circuit.add_gate(H(idx))
            circuit.add_gate(CNOT(0, 1))
            circuit.add_gate(add_T_orTdag(t_id, 1))
            circuit.add_gate(CNOT(0, 1))
            for idx, pauli_id in zip([0, 1], pauli_ids):
                if pauli_id == 1:
                    circuit.add_gate(H(idx))
                elif pauli_id == 2:
                    circuit.add_gate(H(idx))
                    circuit.add_gate(S(idx))
        else:  # CNOT不要
            for idx, pauli_id in zip([0, 1], pauli_ids):
                if pauli_id > 0:
                    position = idx
                if pauli_id == 1:
                    circuit.add_gate(H(idx))
                elif pauli_id == 2:
                    circuit.add_gate(Sdag(idx))
                    circuit.add_gate(H(idx))
            circuit.add_gate(add_T_orTdag(t_id, position))
            for idx, pauli_id in zip([0, 1], pauli_ids):
                if pauli_id == 1:
                    circuit.add_gate(H(idx))
                elif pauli_id == 2:
                    circuit.add_gate(H(idx))
                    circuit.add_gate(S(idx))
    return circuit


def split(pauli_id_lst:list, tf_lst:list)-> list:
    """commuteするもの同士でゲート列を区切る関数

    Args:
        pauli_id_lst (list): パウリ演算子のリスト
        tf_lst (list): True or Falseのリスト

    Returns:
        list: 区切られたリスト
    """
    split_lst = []
    tf_lst.append(True)
    tmp = []
    for i in range(len(pauli_id_lst)):
        tmp.append(pauli_id_lst[i])
        if tf_lst[i] == False:
            split_lst.append(tmp)
            tmp = []
    if len(tmp) != 0:
        split_lst.append(tmp)
    return split_lst


def check_zeros(input_list:list, true_min:int=0, true_max:int=0)->bool:
    """Pauli_idのリストについてフラットなリストに変換してから0の数を数える

    Args:
        input_list (list): Pauli_idのリスト
        true_min (int, optional): 0の個数の最小値. Defaults to 0.
        true_max (int, optional): 0の個数の最大値. Defaults to 0.

    Returns:
        bool: 0の個数がtrue_min以上true_max以下かどうか
    """
    # フラットなリストに変換してから0の数を数える
    flat_list = [item for sublist in input_list for item in sublist]
    zero_count = flat_list.count(0)

    # 0の数が8以上かどうかを確認
    return zero_count >= true_min and zero_count <= true_max


def has_common_pauli_rotation(pauli_id_lst:list)->list:
    """共通のパウリ回転が存在するかどうかを判定する(連続している必要はない)

    Args:
        pauli_id_lst (list): パウリ演算子のリスト

    Returns:
        list: 共通のパウリ回転が存在する場合はそのリスト、存在しない場合はFalse
    """
    s = dict(Counter(pauli_id_lst))
    values = list({key: value for key, value in s.items() if value >= 2}.keys())
    if len(values) > 0:
        return values
    else:
        return False


def has_consecutive(lst:list)->bool:
    """リスト内に連続して同じ要素が存在するかどうかを判定する

    Args:
        lst (list): リスト

    Returns:
        bool: 連続して同じ要素が存在するかどうか
    """
    for i in range(len(lst) - 1):
        if lst[i] == lst[i + 1]:
            return True
    return False


def can_locally_optimize(pauli_ids:list)->bool:
    """局所的に最適化可能かどうかを判定する

    Args:
        pauli_ids (list): パウリ演算子のリスト

    Returns:
        bool: 局所的に最適化可能かどうか
    """
    # print(f'Input: {pauli_ids}')
    target_pauli_ids = has_common_pauli_rotation(pauli_ids)  # 共通の回転軸を持っているか
    if target_pauli_ids == False:
        return False
    else:
        if has_consecutive(pauli_ids):
            return True
        else:
            for target_pauli in target_pauli_ids:
                # print(f'target_pauli: {target_pauli}')
                commutable_pauli_ids = COMMUTE_DICTS[target_pauli]
                # print(f'commutable_pauli_ids: {commutable_pauli_ids}')
                positions = [i for i, ele in enumerate(pauli_ids) if ele == target_pauli]  # 今考えているpauliがいる場所
                for j in range(len(positions) - 1):
                    start, end = positions[j], positions[j + 1]
                    # print(f'start-end: {start}, {end}')
                    checker = [pauli_ids[i] in commutable_pauli_ids for i in range(start + 1, end)]
                    # print(checker)
                    if all(checker) == True:
                        # print(f'start-end: {start}, {end}')
                        return True
            return False