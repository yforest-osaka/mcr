def pauli_bit_to_pauli_id(pauli_bit, with_coef=True):
    sgn_dict = {0: 1, 1: 1j, 2: -1, 3: -1j}
    pauli_bit_to_id_dict = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    if with_coef:
        return sgn_dict[pauli_bit[0]], tuple(
            pauli_bit_to_id_dict[i] for i in pauli_bit[1:]
        )
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
    if isinstance(
        pauli_string, tuple
    ):  # タプルの内部で係数が与えられている場合 (-1, 'XX')
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
