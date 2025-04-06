#  qulacs・pyzxで定義した回路のサンプル置き場
import random

import numpy as np
import pyzx as zx
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, H, ParametricRZ, RotZ, S, T, Tdag  # type: ignore


class QulacsGateSequenceGenerator:
    def __init__(self, qubits: int):
        self.qubits = qubits

    def ccz(self, ancilla: bool = False, qubit_assign: list = [0, 1, 2]) -> list:
        """Generate CCZ (Toffoli) circuit.

        Args:
            ancilla (bool, optional): Whether to add an ancilla qubit. Defaults to False.
            qubit_assign (list, optional): List of qubit indices. Defaults to [0, 1, 2].

        Returns:
            list: List of gates representing the CCZ circuit.
        """
        if ancilla:
            a, b, c, d, e, f, g = qubit_assign
            return [
                CNOT(b, f),
                CNOT(a, d),
                CNOT(b, e),
                CNOT(c, f),
                CNOT(d, g),
                CNOT(a, e),
                CNOT(c, g),
                CNOT(f, d),
                T(a),
                T(b),
                T(c),
                T(d),
                Tdag(e),
                Tdag(f),
                Tdag(g),
                CNOT(f, d),
                CNOT(c, g),
                CNOT(a, e),
                CNOT(d, g),
                CNOT(c, f),
                CNOT(b, e),
                CNOT(a, d),
                CNOT(b, f),
            ]
        else:
            a, b, c = qubit_assign
            return [
                CNOT(b, c),
                Tdag(c),
                CNOT(a, c),
                T(c),
                CNOT(b, c),
                Tdag(c),
                CNOT(a, c),
                Tdag(b),
                T(c),
                CNOT(a, b),
                Tdag(b),
                CNOT(a, b),
                T(a),
                S(b),
            ]

    def toffoli(self, ancilla: bool = False, qubit_assign: list = [0, 1, 2]) -> list:
        """Generate Toffoli circuit.

        Args:
            ancilla (bool, optional): Whether to add an ancilla qubit. Defaults to False.
            qubit_assign (list, optional): List of qubit indices. Defaults to [0, 1, 2].

        Returns:
            list: List of gates representing the Toffoli circuit.
        """
        if ancilla:
            a, b, c, d, e, f, g = qubit_assign
            return [
                H(c),
                CNOT(b, f),
                CNOT(a, d),
                CNOT(b, e),
                CNOT(c, f),
                CNOT(d, g),
                CNOT(a, e),
                CNOT(c, g),
                CNOT(f, d),
                T(a),
                T(b),
                T(c),
                T(d),
                Tdag(e),
                Tdag(f),
                Tdag(g),
                CNOT(f, d),
                CNOT(c, g),
                CNOT(a, e),
                CNOT(d, g),
                CNOT(c, f),
                CNOT(b, e),
                CNOT(a, d),
                CNOT(b, f),
                H(c),
            ]
        else:
            a, b, c = qubit_assign
            return [
                H(c),
                CNOT(b, c),
                Tdag(c),
                CNOT(a, c),
                T(c),
                CNOT(b, c),
                Tdag(c),
                CNOT(a, c),
                Tdag(b),
                T(c),
                CNOT(a, b),
                Tdag(b),
                CNOT(a, b),
                T(a),
                S(b),
                H(c),
            ]

    def toffoli4(self, ancilla: bool = False, qubit_assign: list = [0, 1, 2, 3]) -> list:
        """Generate 4-qubit Toffoli (3 controlled, 1 target) circuit.

        Args:
            ancilla (bool, optional): Whether to add an ancilla qubit. Defaults to False.
            qubit_assign (list, optional): List of qubit indices. Defaults to [0, 1, 2, 3].

        Returns:
            list: List of gates representing the 4-qubit Toffoli circuit.
        """
        if ancilla:
            a, b, c, d, e = qubit_assign
            return [
                H(d),
                H(e),
                RotZ(e, np.pi / 4),
                CNOT(b, e),
                RotZ(e, -np.pi / 4),
                CNOT(a, e),
                RotZ(e, np.pi / 4),
                CNOT(b, e),
                RotZ(e, -np.pi / 4),
                H(e),
                CNOT(e, d),
                RotZ(d, -np.pi / 4),
                CNOT(c, d),
                RotZ(d, np.pi / 4),
                CNOT(e, d),
                RotZ(d, -np.pi / 4),
                RotZ(e, np.pi / 4),
                CNOT(c, d),
                CNOT(c, e),
                RotZ(c, np.pi / 4),
                RotZ(d, np.pi / 4),
                RotZ(e, -np.pi / 4),
                H(d),
                CNOT(c, e),
                H(e),
                RotZ(e, np.pi / 4),
                CNOT(b, e),
                RotZ(e, -np.pi / 4),
                CNOT(a, e),
                RotZ(e, np.pi / 4),
                CNOT(b, e),
                RotZ(e, -np.pi / 4),
                H(e),
            ]
        else:
            a, b, c, d = 0, 1, 2, 3
            return [
                H(d),
                RotZ(a, np.pi / 8),
                RotZ(b, np.pi / 8),
                RotZ(c, np.pi / 8),
                RotZ(d, np.pi / 8),
                CNOT(a, b),
                RotZ(b, -np.pi / 8),
                CNOT(a, b),
                CNOT(b, c),
                RotZ(c, -np.pi / 8),
                CNOT(a, c),
                RotZ(c, np.pi / 8),
                CNOT(b, c),
                RotZ(c, -np.pi / 8),
                CNOT(a, c),
                CNOT(c, d),
                RotZ(d, -np.pi / 8),
                CNOT(b, d),
                RotZ(d, np.pi / 8),
                CNOT(c, d),
                RotZ(d, -np.pi / 8),
                CNOT(a, d),
                RotZ(d, np.pi / 8),
                CNOT(c, d),
                RotZ(d, -np.pi / 8),
                CNOT(b, d),
                RotZ(d, np.pi / 8),
                CNOT(c, d),
                RotZ(d, -np.pi / 8),
                CNOT(a, d),
                H(d),
            ]

    def generate_clifford_circuit(self, depth: int, p_cnot: float = 0.3, p_t: float = 0) -> list:
        """Generate a Clifford circuit using the specified parameters.

        Args:
            depth: The depth of the circuit.
            p_cnot: The probability of applying a CNOT gate. Defaults to 0.3.
            p_t: The probability of applying a T gate. Defaults to 0.

        Returns:
            list: List of gates representing the generated circuit.
        """
        p_s = 0.5 * (1.0 - p_cnot - p_t)
        p_had = 0.5 * (1.0 - p_cnot - p_t)
        n = self.qubits
        gates = []
        for _ in range(depth):
            r = random.random()
            if r > 1 - p_had:
                gates.append(H(random.randrange(n)))
            elif r > 1 - p_had - p_s:
                gates.append(S(random.randrange(n)))
            elif r > 1 - p_had - p_s - p_t:
                gates.append(T(random.randrange(n)))
            else:
                tgt = random.randrange(n)
                while True:
                    ctrl = random.randrange(n)
                    if ctrl != tgt:
                        break
                gates.append(CNOT(tgt, ctrl))
        return gates

    def generate_cnot_rz_circuit(self, depth: int, p_cnot: float = 0.3, p_had: float = 0.2, division: int = 4) -> list:
        """Generate a circuit consisting of CNOT and ParametricRZ gates.

        Args:
            qubits (int): The number of qubits in the circuit.
            depth (int): The depth of the circuit.
            p_cnot (float, optional): The probability of applying a CNOT gate. Defaults to 0.3.
            p_had (float, optional): The probability of applying a Hadamard gate. Defaults to 0.2.
            division (int, optional): The division parameter for the ParametricRZ gate. Defaults to 4.

        Returns:
            list: List of gates representing the generated circuit.
        """
        p_rz = 1.0 - p_cnot - p_had
        qubits = self.qubits
        gates = []
        for _ in range(depth):
            r = random.random()
            if r > 1 - p_had:
                gates.append(H(random.randrange(qubits)))
            elif r > 1 - p_had - p_rz:
                phase = np.pi * random.choice([i for i in range(-1 * division + 1, division + 1) if i != 0]) / division
                gates.append(ParametricRZ(random.randrange(qubits), phase))
            else:
                tgt = random.randrange(qubits)
                while True:
                    ctrl = random.randrange(qubits)
                    if ctrl != tgt:
                        break
                gates.append(CNOT(tgt, ctrl))
        return gates


class PyZXCircuitGenerator:
    def __init__(self, qubits: int):
        """
        Initialize the PyZXCircuitGenerator.

        Args:
            qubits (int): The number of qubits in the circuit.
        """
        self.qubits = qubits

    def generate_clifford_circuit(self, depth: int, p_cnot: float = 0.3, p_t: float = 0) -> zx.Circuit:
        """
        Generate a Clifford circuit using the specified parameters.

        Args:
            depth (int): The depth of the circuit.
            p_cnot (float, optional): The probability of applying a CNOT gate. Defaults to 0.3.
            p_t (float, optional): The probability of applying a T gate. Defaults to 0.

        Returns:
            zx.Circuit: The generated Clifford circuit.
        """
        p_s = 0.5 * (1.0 - p_cnot - p_t)
        p_had = 0.5 * (1.0 - p_cnot - p_t)
        n = self.qubits
        circuit = zx.Circuit(n)
        for _ in range(depth):
            r = random.random()
            if r > 1 - p_had:
                circuit.add_gate("HAD", random.randrange(n))
            elif r > 1 - p_had - p_s:
                circuit.add_gate("S", random.randrange(n))
            elif r > 1 - p_had - p_s - p_t:
                circuit.add_gate("T", random.randrange(n))
            else:
                tgt = random.randrange(n)
                while True:
                    ctrl = random.randrange(n)
                    if ctrl != tgt:
                        break
                circuit.add_gate("CNOT", tgt, ctrl)
        return circuit
