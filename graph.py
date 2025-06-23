from itertools import combinations, product
import numpy as np
from mcr.gate_apply import PauliBit, multiply_all
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import random
import math
from joblib import Parallel, delayed
from time import time
from mcr.filesave import save_by_pickle


def generate_commuting_pairs(nqubit_paulis, pauli_bit_cache):
    commute_pairs = []
    for p1, p2 in combinations(nqubit_paulis, 2):
        if pauli_bit_cache[p1].commutes(pauli_bit_cache[p2]):
            commute_pairs.append((p1, p2))
    return commute_pairs


def is_strongly_regular(G):
    nodes = list(G.nodes)
    v = len(nodes)

    # 1. チェック：次数がすべて同じか
    degrees = [G.degree(n) for n in nodes]
    if len(set(degrees)) != 1:
        print("非正則：次数が一致しません")
        return False
    k = degrees[0]

    # 2. 隣接・非隣接ペアごとに共通隣接数を記録
    lambda_counts = []
    mu_counts = []

    for u, v in combinations(nodes, 2):
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common_neighbors = len(neighbors_u & neighbors_v)

        if G.has_edge(u, v):
            lambda_counts.append(common_neighbors)
        else:
            mu_counts.append(common_neighbors)

    # 3. λ, μ が定数か確認
    if len(set(lambda_counts)) != 1:
        print("λ が一定でない：", set(lambda_counts))
        return False
    if len(set(mu_counts)) != 1:
        print("μ が一定でない：", set(mu_counts))
        return False

    λ = lambda_counts[0]
    μ = mu_counts[0]
    print(f"SRG({len(nodes)}, {k}, {λ}, {μ}) です")
    return True


def check_mcr_candidate(p1, p2, pauli_bit_cache):
    # 同じPauliが入っている場合はスキップ
    if len({p1[0], p1[1], p2[0], p2[1]}) < 4:
        return None

    pb1, pb2 = pauli_bit_cache[p1[0]], pauli_bit_cache[p1[1]]
    pb3, pb4 = pauli_bit_cache[p2[0]], pauli_bit_cache[p2[1]]

    if any(
        [
            pb1.commutes(pb3),
            pb1.commutes(pb4),
            pb2.commutes(pb3),
            pb2.commutes(pb4),
        ]
    ):
        return None

    sgn, pauli_d = multiply_all([pb1, pb2, pb3])
    if pauli_d == pb4.get_pauli_str():
        return (p1, p2)
    return None


def main():
    pauli = ["I", "X", "Y", "Z"]
    nqubits = 4
    nqubit_paulis = ["".join(p) for p in product(pauli, repeat=nqubits)][1:]
    assert len(nqubit_paulis) == 4**nqubits - 1

    # PauliBit キャッシュ
    pauli_bit_cache = {p: PauliBit(p, np.pi / 4) for p in nqubit_paulis}

    # Step 1: commuting pairs の生成
    results = generate_commuting_pairs(nqubit_paulis, pauli_bit_cache)
    print(f"Commuting pairs found: {len(results)}")
    print(f"Number of candidate MCR combinations: {math.comb(len(results), 2)}")
    st = time()
    # Step 2: MCR探索（並列化）
    mcr_candidates = list(combinations(results, 2))
    mcrs = Parallel(n_jobs=5, backend="loky")(
        delayed(check_mcr_candidate)(p1, p2, pauli_bit_cache)
        for p1, p2 in tqdm(mcr_candidates)
    )
    mcrs = [m for m in mcrs if m is not None]
    print(f"Time: {time() - st:.5f} seconds")

    # Step 3: グラフ生成と解析
    G = nx.Graph()
    G.add_edges_from(mcrs)

    num_components = nx.number_connected_components(G)
    print(f"連結成分の数: {num_components}")

    components = list(nx.connected_components(G))
    if not components:
        print("No connected components found.")
        return

    random_component_nodes = random.choice(components)
    subgraph = G.subgraph(random_component_nodes)

    try:
        diameter = nx.diameter(subgraph)
    except nx.NetworkXError:
        diameter = "N/A"
    print(f"グラフの直径（2点間の最長の最短距離）: {diameter}")
    print(f"頂点数: {subgraph.number_of_nodes()}")
    print(f"辺数: {subgraph.number_of_edges()}")
    max_degree = max(dict(G.degree()).values())
    print("最大次数:", max_degree)
    print("強正則グラフか:", is_strongly_regular(subgraph))

    save_by_pickle(f"mcrs_{nqubits}qubit.pickle", mcrs)

    # 描画
    # plt.figure(figsize=(6, 4))
    # nx.draw(
    #     subgraph,
    #     with_labels=True,
    #     node_color="lightblue",
    #     edge_color="gray",
    #     node_size=500,
    # )
    # plt.savefig(f"mcr_graph_{nqubits}.pdf")


if __name__ == "__main__":
    main()
