#!/bin/bash
# $1 = 56/57/58
framework=$1
lr=$2
seed=$3


# (
#     CUDA_VISIBLE_DEVICES=3 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DCRNNx2" $lr $seed &
#     CUDA_VISIBLE_DEVICES=4 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "DySATx2" $lr $seed 
# ) &
# (
#     CUDA_VISIBLE_DEVICES=5 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNHx2" $lr $seed &
#     CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNOx2" $lr $seed 
# ) &&
# (
#     CUDA_VISIBLE_DEVICES=3 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCNx2oGRU" $lr $seed &
#     CUDA_VISIBLE_DEVICES=4 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "GCRNM2x2" $lr $seed 
# ) &
# (
    CUDA_VISIBLE_DEVICES=5 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGATx2" $framework $lr $seed &
#     CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "TGNOptimLx2" $lr $seed 
# ) &&
# CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns-dense/LH10~all~trans~dense_GRUoGCN2x2~16~softplus~1.0e-5~value~-1.sh $lr $seed &&
# CUDA_VISIBLE_DEVICES=6 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/LH10~all~none~16~softplus~1.0e-5~value~-1.sh "EvoGCNOx2" $framework $lr $seed 
echo "DONE"



