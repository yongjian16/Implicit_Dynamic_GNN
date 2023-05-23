/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u dynclass.py \
--model $1 --win-aggr none --source UIHC --target all \
--framework transductive --hidden 16 --activate softplus --epoch 200 \
--lr $2 --weight-decay 1e-5 --clipper value --patience -1 --seed $3 \
--device cuda
