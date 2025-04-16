import copy
import os
import random
import re
from itertools import product
from uuid import uuid4

import numpy as np
import pandas as pd
import pyzx as zx
from mqt import qcec
from mqt.qcec.configuration import augment_config_from_kwargs
from mqt.qcec.pyqcec import Configuration
from qulacs import QuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer as QCO  # type: ignore
from qulacs.gate import (  # type: ignore
    CNOT,
    RZ,
    DenseMatrix,
    H,
    Identity,
    ParametricPauliRotation,
    ParametricRZ,
    RandomUnitary,
    S,
    Sdag,
    T,
    Tdag,
    X,
    Y,
    Z,
)
from qulacs_core import (
    ClsOneControlOneTargetGate,
    ClsOneQubitGate,
    QuantumGate_SingleParameter,
)
from tqdm import tqdm

from mcr.filesave import qulacs_to_qasm
from mcr.mycircuit import GateFactory, MyQuantumProgram, get_gate_info


def assign_gate_id_of_generated_gates(gate_lst, base_id):
    gates = []
    # 新しく生成されたゲートに割り当てるgate idを作る
    base_gate_id = list(base_id)
    for i, gate in enumerate(gate_lst):  # \tilde{Udagger}とAを入れる
        generated_gate_id = tuple(base_gate_id + [i])
        gates.append((generated_gate_id, gate))
    return gates


def get_qubit_count_from_qasm_file(filepath):

    # ファイルを読み込む
    with open(filepath, "r") as file:
        content = file.read()

    # 正規表現でqregの横の数字を取得
    match = re.search(r"qreg q\[(\d+)\];", content)
    if match:
        number = match.group(1)
        return int(number)
    else:
        raise ValueError("qubit数が分かりませんでした")


def equivalence_check_via_mqt_qcec(
    circuit_1, circuit_2, exclude_zx_checker=False, show_log=True
):
    # circuit_1, circuit_2はQASMファイルでもOK
    qubit_limit = 25
    remove_flag1, remove_flag2 = False, False
    if isinstance(circuit_1, QuantumCircuit):
        if circuit_1.get_qubit_count() > qubit_limit:
            print("Skipped: too large qubit")
            return True
        filepath1 = f"qcec_tmp1_{uuid4()}.qasm"
        qulacs_to_qasm(filepath1, circuit_1)
        circuit_1 = filepath1
        remove_flag1 = True
    if isinstance(circuit_2, QuantumCircuit):
        if circuit_2.get_qubit_count() > qubit_limit:
            print("Skipped: too large qubit")
            return True
        filepath2 = f"qcec_tmp2_{uuid4()}.qasm"
        qulacs_to_qasm(filepath2, circuit_2)
        circuit_2 = filepath2
        remove_flag2 = True

    if (
        get_qubit_count_from_qasm_file(circuit_1) > qubit_limit
        and get_qubit_count_from_qasm_file(circuit_2) > qubit_limit
    ):
        print("Skipped: too large qubit")
        return True
    if exclude_zx_checker:
        configuration = Configuration()
        augment_config_from_kwargs(
            configuration,
            {"run_simulation_checker": True, "run_zx_checker": False, "timeout": 5},
        )
        result = qcec.verify(
            circuit_1, circuit_2, configuration=configuration
        ).equivalence
    else:
        result = qcec.verify(circuit_1, circuit_2).equivalence
    if remove_flag1:
        os.remove(circuit_1)
    if remove_flag2:
        os.remove(circuit_2)
    if show_log:
        print(result.name)
    return result.name in {"equivalent", "equivalent_up_to_global_phase"}
