/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u sfhh.py \
--model EvoGCNOx2 --win-aggr none --source SFHH --target all \
--framework inductive --hidden 16 --activate softplus --epoch 1 \
--lr $1 --weight-decay 1e-5 --clipper value --patience -1 --seed $2 \
--device cuda --exp-name 'debug'
