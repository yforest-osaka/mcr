for i in $(seq 2 1 4); do
  qsub -v p=$i mcr_benchmark.sh
done