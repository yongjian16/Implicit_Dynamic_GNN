(CUDA_VISIBLE_DEVICES=2 bash examples/task-train-imdb/imdb/IMP~IMDB~all~trans~none~16~softplus~1.0e-5~value~-1.sh 0.001 56 &
CUDA_VISIBLE_DEVICES=3 bash examples/task-train-imdb/imdb/IMP~IMDB~all~trans~none~16~softplus~1.0e-5~value~-1.sh 0.001 57 &
CUDA_VISIBLE_DEVICES=4 bash examples/task-train-imdb/imdb/IMP~IMDB~all~trans~none~16~softplus~1.0e-5~value~-1.sh 0.001 58) && 
echo "DONE"