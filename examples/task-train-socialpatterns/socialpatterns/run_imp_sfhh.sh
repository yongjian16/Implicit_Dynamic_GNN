framework=$1
lr=$2

(CUDA_VISIBLE_DEVICES=2 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/IMP~SFHH~all~none~16~softplus~1.0e-5~value~-1.sh $framework $lr 56 &
CUDA_VISIBLE_DEVICES=3 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/IMP~SFHH~all~none~16~softplus~1.0e-5~value~-1.sh $framework $lr 57 &
CUDA_VISIBLE_DEVICES=4 bash examples/task-train-socialpatterns/socialpatterns/socialpatterns-none/IMP~SFHH~all~none~16~softplus~1.0e-5~value~-1.sh $framework $lr 58) && 
echo "DONE"