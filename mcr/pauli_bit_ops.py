# Pauli_bitに関連する引数を取るような関数の集まり
# "I": (0, 0), "X": (0, 1), "Y": (1, 1), "Z": (1, 0)} という対応を使っている


def check_tuple_depth(t):
    if isinstance(t, tuple):
        if any(isinstance(i, tuple) for i in t):
            return 2
        else:
            return 1
    else:
        return 0


def pauli_id_to_pauli_bit(pauli_ids):  # 1個のPauli_idを与える
    pauli_id_to_bit_dict = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    if check_tuple_depth(pauli_ids) == 2:  # 係数が与えられている場合
        sgn_dict = {1: 0, 1j: 1, -1: 2, -1j: 3}
        sgn_bit = sgn_dict[pauli_ids[0]]
        pauli_ids = pauli_ids[1]
    else:
        assert check_tuple_depth(pauli_ids) == 1
        sgn_bit = 0  # 係数1
        # pauli_ids = pauli_ids[0]
    results = []
    results.append(sgn_bit)
    for pauli_id in pauli_ids:
        try:
            results.append(pauli_id_to_bit_dict[pauli_id])
        except KeyError:
            raise ValueError(f"Invalid Pauli ID: {pauli_id}")
    return tuple(results)


def pauli_bit_to_pauli_id(pauli_bit, with_coef=True):
    sgn_dict = {0: 1, 1: 1j, 2: -1, 3: -1j}
    pauli_bit_to_id_dict = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    if with_coef:
        return sgn_dict[pauli_bit[0]], tuple(pauli_bit_to_id_dict[i] for i in pauli_bit[1:])
    else:
        return tuple(pauli_bit_to_id_dict[i] for i in pauli_bit[1:])


def pauli_string_to_pauli_bit(pauli_string):  # 1個のPauli_stringを与える
    pauli_str_to_bit_dict = {
        "I": (0, 0),
        "X": (0, 1),
        "Y": (1, 1),
        "Z": (1, 0),
        "_": (0, 0),
    }
    if isinstance(pauli_string, tuple):  # タプルの内部で係数が与えられている場合 (-1, 'XX')
        sgn_dict = {1: 0, 1j: 1, -1: 2, -1j: 3}
        sgn_bit = sgn_dict[pauli_string[0]]
        pauli_string = pauli_string[1]
    else:
        sgn_bit = 0  # 係数1
    if pauli_string[:2] == "+i":  # 先頭が文字列の+iの場合(stimでの利用を想定)
        sgn_bit = 1
        pauli_string = pauli_string[2:]
    elif pauli_string[:2] == "-i":  # 先頭が文字列の-iの場合(stimでの利用を想定)
        sgn_bit = 3
        pauli_string = pauli_string[2:]
    elif pauli_string[0] == "-":  # 先頭が文字列の-の場合(stimでの利用を想定)
        sgn_bit = 2
        pauli_string = pauli_string[1:]
    elif pauli_string[0] == "+":
        sgn_bit = 0
        pauli_string = pauli_string[1:]
    results = []
    results.append(sgn_bit)
    for pauli in pauli_string:
        try:
            results.append(pauli_str_to_bit_dict[pauli.upper()])
        except KeyError:
            raise ValueError(f"Invalid Pauli String: {pauli}")
    return tuple(results)


def pauli_bit_to_pauli_string(pauli_bit, with_coef=True):
    sgn_dict = {0: 1, 1: 1j, 2: -1, 3: -1j}
    pauli_bit_to_id_dict = {(0, 0): "I", (0, 1): "X", (1, 1): "Y", (1, 0): "Z"}
    data = [pauli_bit_to_id_dict[i] for i in pauli_bit[1:]]
    if with_coef:
        return sgn_dict[pauli_bit[0]], "".join(data)
    else:
        return "".join(data)


def one_pauli_mult(pauli_bit_1, pauli_bit_2):
    # Pauli行列の符号と組み合わせに基づくルックアップ辞書
    pauli_signs = {
        ((0, 1), (1, 1)): 1,  # X @ Y
        ((1, 1), (1, 0)): 1,  # Y @ Z
        ((1, 0), (0, 1)): 1,  # Z @ X
        ((1, 1), (0, 1)): 3,  # Y @ X
        ((1, 0), (1, 1)): 3,  # Z @ Y
        ((0, 1), (1, 0)): 3,  # X @ Z
    }

    # XORによる新しい値の計算
    new_val = tuple(i ^ j for i, j in zip(pauli_bit_1, pauli_bit_2))

    # sgnの取得（該当する組み合わせがない場合は0）
    sgn = pauli_signs.get((pauli_bit_1, pauli_bit_2), 0)

    return sgn, new_val


def multiply_pauli_bits(pauli_bits_a, pauli_bits_b):
    sgn = pauli_bits_a[0] + pauli_bits_b[0]
    results = []
    for i in range(1, len(pauli_bits_a)):
        new_sgn, new_val = one_pauli_mult(pauli_bits_a[i], pauli_bits_b[i])
        sgn += new_sgn
        results.append(new_val)
    return (sgn % 4, *results)


def expand(left, right):
    results = []
    left_pauli_bits_all = [pauli_id_to_pauli_bit(ele) for ele in left]
    right_pauli_bits_all = [pauli_id_to_pauli_bit(ele) for ele in right]
    for left_ele in left_pauli_bits_all:
        for right_ele in right_pauli_bits_all:
            results.append(multiply_pauli_bits(left_ele, right_ele))
    assert len(results) == len(left) * len(right)
    return results


def sgn_flip(pauli_bits_lst):  # 係数だけマイナス1倍する(引き算を実装した時に使う)
    results = []
    for ele in pauli_bits_lst:
        results.append(((ele[0] + 2) % 4, *ele[1:]))
    return results


# 0,1,2,3のdictを作る。keyがpauli_id、valueがそのpauli_idの個数
def check_zero(pauli_bits_lst):
    zero_dicts, one_dicts, two_dicts, three_dicts = {}, {}, {}, {}

    for ele in pauli_bits_lst:
        pauli = tuple(ele[1:])
        if ele[0] == 0:
            zero_dicts[pauli] = zero_dicts.get(pauli, 0) + 1
        elif ele[0] == 1:
            one_dicts[pauli] = one_dicts.get(pauli, 0) + 1
        elif ele[0] == 2:
            two_dicts[pauli] = two_dicts.get(pauli, 0) + 1
        elif ele[0] == 3:
            three_dicts[pauli] = three_dicts.get(pauli, 0) + 1

    # 各辞書を比較して、違いがあれば False を返す
    return zero_dicts == two_dicts and one_dicts == three_dicts


def pauli_bit_commute(left_pauli_id_lst, right_pauli_id_lst):
    if isinstance(left_pauli_id_lst, tuple):
        left_pauli_id_lst = [left_pauli_id_lst]
    if isinstance(right_pauli_id_lst, tuple):
        right_pauli_id_lst = [right_pauli_id_lst]
    commutator = expand(left_pauli_id_lst, right_pauli_id_lst) + sgn_flip(expand(right_pauli_id_lst, left_pauli_id_lst))
    return check_zero(commutator)
