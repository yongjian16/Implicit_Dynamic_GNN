/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u pems.py \
--model $1 --source PeMS08 --target all --framework $2 \
--hidden 16 --activate softplus --epoch 100 --lr 1e-3 --weight-decay 1e-5 \
--clipper value --patience -1 --seed $3 --device cuda
