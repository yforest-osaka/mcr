#!/bin/bash
# directory="/home/yusei/work/unoptimization/dataset_1222/"

# cd /home/yusei/work/unoptimization/quantum-circuit-optimization

# for qc_file in "$directory"/v*.qc; do
#     cargo +nightly run -r "$qc_file" &
# done
# wait

# ====================

directory="/Users/yusei/Desktop/workspace/d1_work/mcr/tmp"

cd /Users/yusei/Desktop/workspace/d1_work/mcr/quantum-circuit-optimization

# findコマンドを使って、再帰的にすべての.qcファイルを取得
find "$directory" -type f -name "*.qc" | while read -r qc_file; do
    cargo +nightly run -r "$qc_file" &
done

