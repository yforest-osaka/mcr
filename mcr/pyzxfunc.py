import io
import sys
from collections import Counter

import pyzx as zx
from pyzx.graph.graph_s import GraphS
from qulacs import QuantumCircuit
from qulacs.converter import convert_qulacs_circuit_to_QASM


def optimize(pyzx_circuit: zx.circuit.Circuit, quiet: bool = True) -> zx.circuit.Circuit:
    """PyZXを使った最適化(max)

    Args:
        pyzx_circuit (Circuit): PyZXのCircuit
        quiet (bool, optional): ログを出力するか. Defaults to True.

    Returns:
        Circuit: 最適化されたCircuit
    """
    g = pyzx_circuit.to_graph()
    zx.full_reduce(g, quiet=quiet)  # simplifies the Graph in-place, and show the rewrite steps taken.
    g.normalize()  # Makes the graph more suitable for displaying
    # zx.draw(g) # Display the resulting diagram
    c_opt = zx.extract_circuit(g.copy())
    # if c_opt.tcount()>0:
    #     print(c_opt.tcount())
    return c_opt


def graph_optimize(g: GraphS, quiet: bool = True) -> GraphS | zx.circuit.Circuit:
    """PyZXのグラフのfull_reduce optimize(アンシラあり回路用)

    Args:
        g (GraphS): PyZXのGraph
        quiet (bool, optional): ログを出力するか. Defaults to True.

    Returns:
        GraphS | zx.circuit.Circuit: 最適化されたGraph or Circuit
    """

    zx.full_reduce(g, quiet=quiet)  # simplifies the Graph in-place, and show the rewrite steps taken.(強い...?)
    # zx.optimize.full_optimize(circuit_data[i])
    g.normalize()  # Makes the graph more suitable for displaying
    # zx.draw(g) # Display the resulting diagram
    try:
        c_opt = zx.extract_circuit(g.copy())
        return c_opt
    except:
        return g


def pyzx_compiler(initial_circuit: QuantumCircuit, unopted_circuit: QuantumCircuit) -> tuple[float, float]:
    """PyZXを使った非最適化→最適化

    Args:
        initial_circuit (QuantumCircuit): Qulacsの回路
        unopted_circuit (QuantumCircuit): Qulacsの回路

    Returns:
        tuple[float, float]: 最適化前のtcount比, 最適化後のtcount比
    """
    from mcr.filesave import qasm_to_pyzx, qulacs_to_qasm

    qulacs_to_qasm("tmp_u.qasm", initial_circuit)
    qulacs_to_qasm("tmp_v.qasm", unopted_circuit)
    initial_circuit_pyzx = qasm_to_pyzx("tmp_u.qasm")
    unopted_circuit_pyzx = qasm_to_pyzx("tmp_v.qasm")
    opt_initial_circuit_pyzx = optimize(initial_circuit_pyzx)
    t_initial = opt_initial_circuit_pyzx.tcount()

    before_optimized_circuit = unopted_circuit_pyzx.copy()
    t_unopted = before_optimized_circuit.tcount()

    opt_unopted_circuit_pyzx = optimize(unopted_circuit_pyzx)
    t_opted = opt_unopted_circuit_pyzx.tcount()
    r_unopt = t_unopted / t_initial
    r_opt = t_opted / t_initial
    return r_unopt, r_opt


def find_duplicate_tuples(tuple_list: list[tuple]) -> list:
    """リスト内のタプルについて、出現回数が2回以上のタプルを抽出

    Args:
        tuple_list (list[tuple]): タプルのリスト

    Returns:
        list: 重複したタプルのリスト
    """

    tuple_count = Counter(tuple_list)

    duplicates = [item for item, count in tuple_count.items() if count > 1]

    return duplicates


def capture_log(func, *args, **kwargs):
    """PyZXの最適化手法を実行し、ログを取得する

    Args:
        func (_type_): PyZXの最適化手法を入れる

    Returns:
        _type_: _description_
    """
    # Redirect stdout
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Run the function
    result = func(*args, **kwargs)

    # Capture the output
    output = new_stdout.getvalue()

    # Restore stdout
    sys.stdout = old_stdout

    return result, output


def log_arrange(log_data: str) -> list:
    """PyZXのログを整形する

    Args:
        log_data (str): PyZXのログ(文字列)

    Returns:
        list: 整形されたログ
    """
    log_lines = log_data.split("\n")
    log_list = [line.strip() for line in log_lines if line.strip()]
    log_list = [elem.split(":")[0] for elem in log_list]
    return log_list


def show_process(zx_graph: GraphS, log_list: list) -> GraphS:
    """PyZXの最適化プロセスを可視化する

    Args:
        zx_graph (GraphS): PyZXのGraph
        log_list (list): PyZXのログ

    Raises:
        ValueError: 最適化の種類が不正な場合

    Returns:
        GraphS: 最適化後のGraph
    """
    zx.to_gh(zx_graph)
    zx.draw(zx_graph)
    print("------")
    for elem in log_list:
        if elem == "spider_simp":
            zx.simplify.spider_simp(zx_graph, quiet=False)
        elif elem == "id_simp":
            zx.simplify.id_simp(zx_graph, quiet=False)
        elif elem == "pivot_simp":
            zx.simplify.pivot_simp(zx_graph, quiet=False)
        elif elem == "lcomp_simp":
            zx.simplify.lcomp_simp(zx_graph, quiet=False)
        elif elem == "pivot_gadget_simp":
            zx.simplify.pivot_gadget_simp(zx_graph, quiet=False)
        elif elem == "pivot_boundary_simp":
            zx.simplify.pivot_boundary_simp(zx_graph, quiet=False)
        elif elem == "gadget_simp":
            zx.simplify.gadget_simp(zx_graph, quiet=False)
        else:
            raise ValueError(f"{elem}")
        zx_graph.normalize()
        zx.draw(zx_graph)
        print("------")
        # input()
    return zx_graph


def check_tcount_after_simp(circ: zx.circuit.Circuit, get_graph: bool = False) -> tuple:
    """最適化後のtcountを取得する

    Args:
        circ (zx.circuit.Circuit): PyZXのCircuit
        get_graph (bool, optional): グラフを取得するか. Defaults to False.

    Returns:
        tuple: 最適化後のtcount, 左端のboundary spiderの数, 右端のboundary spiderの数
    """
    # spider_simpとid_simpをできるところまで繰り返した後のtcount
    id_graph = circ.to_graph()
    zx.to_gh(id_graph)
    t_flag = True
    while t_flag:
        a = zx.simplify.spider_simp(id_graph, quiet=True)
        b = zx.simplify.id_simp(id_graph, quiet=True)
        c = zx.simplify.spider_simp(id_graph, quiet=True)
        d = zx.simplify.id_simp(id_graph, quiet=True)
        if a + b + c + d == 0:
            t_flag = False
            tmp_c = zx.extract_circuit(id_graph.copy())
            boundary_tcount_l, boundary_tcount_r = get_boundary_tcount(id_graph.copy())
            if get_graph:
                id_graph.normalize()
                return tmp_c.tcount(), boundary_tcount_l, boundary_tcount_r, id_graph
            else:
                return tmp_c.tcount(), boundary_tcount_l, boundary_tcount_r


def get_boundary_tcount(pyzx_graph: GraphS) -> tuple:
    """境界部分にあるTゲートの数を取得する

    Args:
        pyzx_graph (GraphS): PyZXのGraph

    Returns:
        tuple: 左端のboundary Tゲートの数, 右端のboundary Tゲートの数
    """
    tmp = pyzx_graph.copy()
    # incident_edgesでboundary spider4個をなんとか引き出してくる
    v_count = tmp.num_vertices()
    tmp_vertices = [tmp.incident_edges(i)[0][1] for i in [0, 1]]
    boundary_counter_l = 0
    for vertex_number in tmp_vertices:
        if tmp.phase(vertex_number).denominator == 4:
            boundary_counter_l += 1

    boundary_counter_r = 0
    tmp_vertices = [tmp.incident_edges(i)[0][0] for i in [v_count - 1, v_count - 2]]
    for vertex_number in tmp_vertices:
        if tmp.phase(vertex_number).denominator == 4:
            boundary_counter_r += 1
    return boundary_counter_l, boundary_counter_r
