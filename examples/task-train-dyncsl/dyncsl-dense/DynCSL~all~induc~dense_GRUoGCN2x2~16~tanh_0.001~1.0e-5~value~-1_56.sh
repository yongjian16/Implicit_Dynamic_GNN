/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u dynclass.py \
--model GRUoGCN2x2 --win-aggr dense --source DynCSL --target all \
--framework inductive --hidden 16 --activate tanh --epoch 5 --lr 1e-3 \
--weight-decay 1e-5 --clipper value --patience -1 --seed 56 --device cuda
