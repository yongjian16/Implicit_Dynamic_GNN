(CUDA_VISIBLE_DEVICES=3 bash examples/task-train-brain10/brain10-none/IMP~Brain10~all~trans~none~16~softplus~1.0e-5~value~-1.sh 0.01 56 &
CUDA_VISIBLE_DEVICES=4 bash examples/task-train-brain10/brain10-none/IMP~Brain10~all~trans~none~16~softplus~1.0e-5~value~-1.sh 0.01 58 & 
CUDA_VISIBLE_DEVICES=5 bash examples/task-train-brain10/brain10-none/IMP~Brain10~all~trans~none~16~softplus~1.0e-5~value~-1.sh 0.01 57)
&& 
echo "DONE"
