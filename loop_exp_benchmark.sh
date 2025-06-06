for i in $(seq 2 1 3); do
  qsub -v p=$i exp_benchmark.sh
done