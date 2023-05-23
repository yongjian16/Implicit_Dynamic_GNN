/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u imp_pems.py \
--model IDGNN --source PeMS04 --target all --framework $1 \
--hidden 16 --activate softplus --epoch 100 --lr $2 --weight-decay 1e-5 \
--clipper value --patience -1 --seed $3 --device cuda
