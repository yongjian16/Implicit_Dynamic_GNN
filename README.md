## Usage

Data is provided by [Dropbox](https://www.dropbox.com/s/yo74hwj7k8alpuj/ICML2022-Data-Submit.zip?dl=0).
Please download and unzip it under `src` directory.
For example, `src/DynCSL/tgnn-power-v2.pt` should a valid path.

We provide example code under `examples` directory.
For example, `examples/task-train-dyncsl/dyncsl-dense\DynCSL~all~induc~dense_GRUoGCN2x2~16~tanh_0.001~1.0e-5~value~-1_56.sh` provide the script to train GRU-GCN on DynCSL dataset.
You can modify the number of epochs by changing `--epochs 1` to specific value.
You can also modify other hyperparameters by changing corresponding arguments.
If run successfully, it will create a pytorch saving file with exactly the same name as the script with extension `.ptnnp` under `log` directory.

The codes are implemented with CUDA 10.2, and requires `pytorch_geometric` and `more_itertools`.
