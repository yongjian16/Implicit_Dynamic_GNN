python -u sfhh.py ^
--model EvoGCNOx2 --win-aggr dense --source SFHH --target all ^
--framework inductive --hidden 16 --activate softplus --epoch 1 ^
--lr 0.01 --weight-decay 1e-5 --clipper value --patience -1 --seed 56 ^
--device cuda --exp-name 'debug'
