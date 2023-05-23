/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u imp_dynclass.py \
--model IDGNN --win-aggr none --source Reddit4 --target all \
--framework transductive --hidden 16 --activate softplus --epoch 500 \
--lr $1 --weight-decay 1e-5 --clipper value --patience -1 --seed $2 \
--device cuda --exp-name 'exp1'