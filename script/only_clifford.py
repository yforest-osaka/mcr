from typing import List
from pathlib import Path
from tqdm import tqdm


def filter_qasm_list(qasm_lines: List[str]) -> List[str]:
    """
    QASMコードの各行が要素となっているリストから、't 'または'tdg 'で始まる行を除外する。

    Parameters:
    - qasm_lines (List[str]): QASMコード行のリスト

    Returns:
    - List[str]: フィルター後のQASMコード行リスト
    """
    filtered_lines = []
    for line in qasm_lines:
        stripped = line.strip()
        if stripped.startswith("t ") or stripped.startswith("tdg "):
            continue
        filtered_lines.append(line)
    return filtered_lines


def main():
    target_dir = Path("../external_data/qasm")
    for filename in tqdm(target_dir.glob("*.qasm"), desc="Processing files"):
        with open(f"../external_data/qasm/{filename.name}", "r") as f:
            string = f.readlines()
        circ_qasm = [line for line in string if line != ""]
        result = filter_qasm_list(circ_qasm)
        with open(f"../external_data/only_clifford/{filename.name}", "w") as f:
            for line in result:
                f.write(line)


if __name__ == "__main__":
    main()
    print("done!")
