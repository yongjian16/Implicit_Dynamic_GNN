#!/bin/bash
# $1 = 56/57/58
framework=$1
lr=$2

(
    CUDA_VISIBLE_DEVICES=0 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNOx2" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=1 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNOx2" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=2 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNOx2" $framework $lr 58 &

    CUDA_VISIBLE_DEVICES=3 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCNx2oGRU" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCNx2oGRU" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCNx2oGRU" $framework $lr 58 &
    
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGNOptimLx2" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGNOptimLx2" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=7 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGNOptimLx2" $framework $lr 58 &
) && (
    CUDA_VISIBLE_DEVICES=0 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNHx2" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=1 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNHx2" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=2 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNHx2" $framework $lr 58 &

    CUDA_VISIBLE_DEVICES=3 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGATx2" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGATx2" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGATx2" $framework $lr 58 &

    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns-none/IMP~LH10~all~none~16~softplus~1.0e-5~value~-1.sh $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=7 bash examples/task-train-socialpatterns/socialpatterns-none/IMP~LH10~all~none~16~softplus~1.0e-5~value~-1.sh $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=7 bash examples/task-train-socialpatterns/socialpatterns-none/IMP~LH10~all~none~16~softplus~1.0e-5~value~-1.sh $framework $lr 58 &

) && (
    CUDA_VISIBLE_DEVICES=0 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DCRNNx2" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=1 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DCRNNx2" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=2 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DCRNNx2" $framework $lr 58 &

    CUDA_VISIBLE_DEVICES=3 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DySATx2" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=4 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DySATx2" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DySATx2" $framework $lr 58 &

    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCRNM2x2" $framework $lr 56 &
    CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCRNM2x2" $framework $lr 57 &
    CUDA_VISIBLE_DEVICES=7 bash examples/task-train-socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCRNM2x2" $framework $lr 58 &

) &&

# (
#     CUDA_VISIBLE_DEVICES=5 bash examples/task-train-socialpatterns/socialpatterns-dense/LH10~all~dense~16~softplus~1.0e-5~value~-1.sh $framework $lr 56 &
#     CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns-dense/LH10~all~dense~16~softplus~1.0e-5~value~-1.sh $framework $lr 57 &
#     CUDA_VISIBLE_DEVICES=7 bash examples/task-train-socialpatterns/socialpatterns-dense/LH10~all~dense~16~softplus~1.0e-5~value~-1.sh $framework $lr 58 &
# ) &&
echo "DONE"



