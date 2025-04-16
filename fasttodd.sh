#!/bin/bash

directory="/home/yusei/m2_work/unoptimization/with_swap_dataset_1222/n=8_k=64/qc" # qcファイルのデータが入っているフォルダを指定

cd ./quantum-circuit-optimization # fasttoddをインストールしているのディレクトリに移動


# cargo +nightly run -r "../input_data/qc/v_n_4_k_16_0.qc" &
cargo +nightly run -r "/Users/yusei/Desktop/workspace/d1_work/mcr/tmp.qc" &