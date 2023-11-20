framework=$1
lr=$2
seed=$3

/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u socialpatterns.py \
--model GRUoGCN2x2 --win-aggr dense --source SFHH --target all \
--framework $framework --hidden 16 --activate softplus --epoch 100 \
--lr $lr --weight-decay 1e-5 --clipper value --patience -1 --seed $seed \
--device cuda --exp-name 'batching' --resume-eval log