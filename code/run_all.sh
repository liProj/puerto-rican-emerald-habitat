#!/usr/bin/env bash
# Remaining experiment blocks. Two GPU queues run concurrently; the CPU baseline queue is
# separate. Each block writes its own CSV under Birds/results/ and can be resumed independently.
set -u
cd /home/snakehand/Documents/mdpi_papers
PY=./.venv/bin/python
L=Birds/logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4

queue_gpu_a() {                       # headline result, then the stack-level ablation
  $PY Birds/code/run_main.py main_spatial --only "HERON" --reps 2 --suffix "_heron" \
      > $L/run_spatial_heron.log 2>&1
  $PY Birds/code/oof_predictions.py                > $L/run_oof.log 2>&1
  $PY Birds/code/final_model.py                    > $L/run_final_model.log 2>&1
  $PY Birds/code/run_main.py ablation_stack --reps 1 > $L/run_ablation_stack.log 2>&1
}
queue_gpu_b() {                       # neural-branch ablation grid, then the cheaper protocols
  $PY Birds/code/run_main.py ablation_nn --reps 2   > $L/run_ablation_nn.log 2>&1
  $PY Birds/code/run_main.py temporal --reps 3      > $L/run_temporal.log 2>&1
  $PY Birds/code/run_main.py main_random --reps 3   > $L/run_random.log 2>&1
  $PY Birds/code/run_main.py main_evi2 --reps 1     > $L/run_evi2.log 2>&1
  $PY Birds/code/run_main.py blocksize --reps 1     > $L/run_blocksize.log 2>&1
}
case "${1:-all}" in
  a) queue_gpu_a ;;
  b) queue_gpu_b ;;
  *) queue_gpu_a & queue_gpu_b & wait ;;
esac
echo "run_all.sh: $1 finished"
