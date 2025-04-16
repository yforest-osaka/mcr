import numpy as np
import sympy as sp
from IPython.display import Math, display
from qulacs import QuantumCircuit
from qulacs.gate import SWAP, H, Identity, S, Sdag, T, Tdag, X, Y, Z  # type: ignore


def complement_identity(circuit: QuantumCircuit) -> QuantumCircuit:
    """一個もゲートがないqubit_indexが存在する場合飲みIdentityゲートを追加する

    Args:
        circuit (QuantumCircuit): Qulacsの回路

    Returns:
        QuantumCircuit: Identityゲートを追加した回路
    """
    indices = []
    for i in range(circuit.get_gate_count()):
        indices += circuit.get_gate(i).get_target_index_list()
        indices += circuit.get_gate(i).get_control_index_list()
    for num in range(circuit.get_qubit_count()):
        if num not in indices:
            circuit.add_gate(Identity(num))
    return circuit
