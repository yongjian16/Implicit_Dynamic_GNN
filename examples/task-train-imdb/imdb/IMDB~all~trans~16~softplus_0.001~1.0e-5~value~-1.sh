/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u dynclass.py \
--model $1 --win-aggr none --source IMDB --target all \
--framework transductive --hidden 16 --activate softplus --epoch 200 \
--lr 1e-3 --weight-decay 1e-5 --clipper value --patience -1 --seed $2 \
--device cuda
