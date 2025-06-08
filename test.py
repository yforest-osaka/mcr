import subprocess
import os
from pathlib import Path

# 処理対象のディレクトリと作業ディレクトリを定義
directory = "/Users/yusei/Desktop/workspace/d1_work/mcr/tmp"
workdir = "/Users/yusei/Desktop/workspace/d1_work/mcr/quantum-circuit-optimization"

# 作業ディレクトリに移動
os.chdir(workdir)

# .qc ファイルを再帰的に取得して実行
qc_files = Path(directory).rglob("*.qc")

processes = []
for qc_file in qc_files:
    print(f"Running cargo for {qc_file}")
    # 非同期で実行（バックグラウンド相当）
    proc = subprocess.Popen(
        ["cargo", "+nightly", "run", "-r", str(qc_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(proc)

# 全てのプロセスが完了するまで待機（必要に応じて）
for proc in processes:
    stdout, stderr = proc.communicate()
    print(stdout.decode())
    if stderr:
        print(stderr.decode())
