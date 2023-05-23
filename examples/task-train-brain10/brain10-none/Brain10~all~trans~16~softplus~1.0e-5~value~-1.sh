/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u dynclass.py \
--model $1 --win-aggr none --source Brain10 --target all \
--framework transductive --hidden 16 --activate softplus --epoch 300 \
--lr $2 --weight-decay 1e-5 --clipper value --patience 100 --seed $3 \
--device cuda --exp-name 'no_normalized'
