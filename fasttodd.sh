#!/bin/bash

directory="/home/yusei/m2_work/unoptimization/with_swap_dataset_1222/n=8_k=64/qc" # qcファイルのデータが入っているフォルダを指定

cd /home/yusei/m2_work/unoptimization/quantum-circuit-optimization # fasttoddをインストールしているのディレクトリに移動

# findコマンドを使って、再帰的にすべての.qcファイルを取得
find "$directory" -type f -name "*.qc" | while read -r qc_file; do
    cargo +nightly run -r "$qc_file" &
done

wait