/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u dynclass.py \
--model GRUoGCN2x2 --win-aggr dense --source Brain10 --target all \
--framework transductive --hidden 16 --activate softplus --epoch 200 \
--lr 1e-3 --weight-decay 1e-5 --clipper value --patience -1 --seed $1 \
--device cpu
