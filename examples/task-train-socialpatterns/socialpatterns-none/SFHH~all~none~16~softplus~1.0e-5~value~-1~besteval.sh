model=$1
framework=$2
lr=$3
seed=$4

/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u socialpatterns.py \
--model $model --win-aggr none --source SFHH --target all \
--framework $framework --hidden 16 --activate softplus --epoch 100 \
--lr $lr --weight-decay 1e-5 --clipper value --patience -1 --seed $seed \
--device cuda --exp-name 'exp1'  --resume-eval log
