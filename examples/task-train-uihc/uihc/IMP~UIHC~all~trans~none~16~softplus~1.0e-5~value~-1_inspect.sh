/usr/bin/time -f "Max CPU Memory: %M KB\nElapsed: %e sec" \
python -u imp_dynclass.py \
--model IDGNN --win-aggr none --source UIHC --target all \
--framework transductive --hidden 16 --activate softplus --epoch 500 \
--lr 0.01 --weight-decay 1e-5 --clipper value --patience 50 --seed 56 \
--device cuda --exp-name 'exp1' --resume-eval "log/"
