import torch
from glob import glob
import numpy as np 
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from typing import cast

'''
Clasification metrics: 
    return [
        (len(ce), torch.sum(ce).item()),
        (len(err), torch.sum(err).item()),
        (1, -macro),
    ]

Regression metrics: 
    return (
        [
            (len(mse), torch.sum(mse).item()),
            (len(rmse), torch.sum(rmse).item()),
            (len(mape), torch.sum(mape).item()),
        ]
    )

Saved information 
    torch.save(
        (
            self.factors, logs, metrics_valid[validon], metrics_test,
            gpu_mem_peak, timecosts,
        ),
        self.ptlog,
    )

    torch.save(
        (
            factors, logs, metrics_valid[validon], metrics_test,
            gpu_mem_peak, timecosts,
        ),
        self.ptbev,
    )

    torch.save(self.neuralnet.state_dict(), self.ptnnp)
'''
def plot_curves(dataset, framework, metric, task, model_name, dynedge = None, lr=None, wd='1e-5', patience=-1, seed = '*', rootdir='log', target = 'all', exp_name=''):
    # take data from ptlog file and plot metric curves
    # EngCOVID~all~induc~none_GCNx2oGRU~16~softplus_0.001~1.0e-5~value~-1_57.ptres
    cls_metric_to_index = {
        'CE': 0, 'ERR': 1, 'ROCAUC': 2
    }
    reg_metric_to_index = {
        'MSE': 0, 'RMSE': 1, 'MAPE': 2
    }
    if task == 'cls':
        m2i = cls_metric_to_index
    elif task == 'reg':
        m2i  = reg_metric_to_index
    else: 
        print('invalid task')

    res_seeds = []
    # fname_pattern = (f"{dataset}~{target}~{framework}" + 
    #                 (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
    #                 f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}.ptlog")
    # if len(exp_name) > 0:
    #     fname_pattern = (f"{dataset}~{target}~{framework}" + 
    #                 (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
    #                 f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}~{exp_name}.ptlog")
    # fpath_pattern = str(Path(rootdir) / fname_pattern)
    # all_fpaths = glob(fpath_pattern)
    # print('pattern: ', fpath_pattern)
    if dynedge is None:
        if seed == '*' or isinstance(seed, int):
            fname_pattern = (f"{dataset}~{target}~{framework}" + 
                            (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                            f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}.ptlog")
            if len(exp_name) > 0:
                fname_pattern = (f"{dataset}~{target}~{framework}" + 
                            (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                            f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}~{exp_name}.ptlog")
            fpath_pattern = str(Path(rootdir) / fname_pattern)
            all_fpaths = glob(fpath_pattern)
        elif isinstance(seed, list):
            all_fpaths = []
            for s in seed:
                fname_pattern = (f"{dataset}~{target}~{framework}" + 
                            (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                            f"~16~softplus_{lr}~{wd}~value~{patience}_{s}.ptlog")
                if len(exp_name) > 0:
                    fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{s}~{exp_name}.ptlog")
                fpath_pattern = str(Path(rootdir) / fname_pattern)
                all_fpaths += glob(fpath_pattern)
    else:
        if seed == '*' or isinstance(seed, int):
            fname_pattern = (f"{dataset}~{target}~{framework}" + 
                            (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                            f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}.ptlog")
            if len(exp_name) > 0:
                fname_pattern = (f"{dataset}~{target}~{framework}" + 
                            (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                            f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}~{exp_name}.ptlog")
            fpath_pattern = str(Path(rootdir) / fname_pattern)
            all_fpaths = glob(fpath_pattern)
        elif isinstance(seed, list):
            all_fpaths = []
            for s in seed:
                fname_pattern = (f"{dataset}~{target}~{framework}" + 
                            (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                            f"~16~softplus_{lr}~{wd}~value~{patience}_{s}.ptlog")
                if len(exp_name) > 0:
                    fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{s}~{exp_name}.ptlog")
                fpath_pattern = str(Path(rootdir) / fname_pattern)
                all_fpaths += glob(fpath_pattern)
        
    if len(all_fpaths) == 0:
        print('no files found with pattern: ', fpath_pattern)
        return
    print(f'found {len(all_fpaths)} files')

    for fpath in all_fpaths:
        data = torch.load(fpath)
        metrics = torch.tensor(data[1]).numpy()
        value = metrics[:, m2i[metric]]
        if metric == 'ROCAUC':
            value = -value
        res_seeds.append(value)
    
    # plot curves for each seed
    for i, seed in enumerate(res_seeds):
        plt.plot(seed, label=f'seed {i}')
    plt.legend()
    # {metric} of {framework} task on {dataset}
    plt.title(f'{model_name} - {framework} task, lr={lr:.4f} on {dataset}')
    plt.xlabel('epoch')
    plt.ylabel(metric)
    savepath = fpath_pattern.replace('.ptlog', f'~{metric}.png')
    plt.savefig(savepath)
    print(f'saved to {savepath}')
    plt.close()



def parse_logs(dataset, framework, metric, task, model_names, dynedge = None, lr=None, 
               wd='1e-5', patience=-1, seed = '*', rootdir='log', target = 'all', exp_name = '', besteval=False):
    '''
    dataset: Brain10, EngCOVID
    framework: trans, induc
    dynedge: dense, none

    '''
    cls_metric_to_index = {
        'CE': 0, 'ERR': 1, 'ROCAUC': 2
    }
    reg_metric_to_index = {
        'MSE': 0, 'RMSE': 1, 'MAPE': 2
    }
    if task == 'cls':
        m2i = cls_metric_to_index
    elif task == 'reg':
        m2i  = reg_metric_to_index
    else: 
        print('invalid task')
    
    # EngCOVID~all~induc~none_GCNx2oGRU~16~softplus_0.001~1.0e-5~value~-1_57.ptres
    testset = {}
    valset = {}
    memories = {}
    timecosts = {}
    if besteval:
        file_ext = 'ptbev'
    else:
        file_ext = 'ptlog'
    for model_name in model_names:
        res_test_seeds = []
        res_val_seeds = []
        memories_seeds = []
        timecosts_seeds = []
        if dynedge is None:
            if seed == '*' or isinstance(seed, int):
                fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}.{file_ext}")
                if len(exp_name) > 0:
                    fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}~{exp_name}.{file_ext}")
                fpath_pattern = str(Path(rootdir) / fname_pattern)
                all_fpaths = glob(fpath_pattern)
            elif isinstance(seed, list):
                all_fpaths = []
                for s in seed:
                    fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{s}.{file_ext}")
                    if len(exp_name) > 0:
                        fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                    (f"_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                    f"~16~softplus_{lr}~{wd}~value~{patience}_{s}~{exp_name}.{file_ext}")
                    fpath_pattern = str(Path(rootdir) / fname_pattern)
                    all_fpaths += glob(fpath_pattern)
        else:
            if seed == '*' or isinstance(seed, int):
                fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}.{file_ext}")
                if len(exp_name) > 0:
                    fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{seed}~{exp_name}.{file_ext}")
                fpath_pattern = str(Path(rootdir) / fname_pattern)
                all_fpaths = glob(fpath_pattern)
            elif isinstance(seed, list):
                all_fpaths = []
                for s in seed:
                    fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                f"~16~softplus_{lr}~{wd}~value~{patience}_{s}.{file_ext}")
                    if len(exp_name) > 0:
                        fname_pattern = (f"{dataset}~{target}~{framework}" + 
                                    (f"~{dynedge}_{model_name}" if dynedge is not None else f"_{model_name}") + 
                                    f"~16~softplus_{lr}~{wd}~value~{patience}_{s}~{exp_name}.{file_ext}")
                    fpath_pattern = str(Path(rootdir) / fname_pattern)
                    all_fpaths += glob(fpath_pattern)
        
        # print('pattern: ', fpath_pattern)
        if len(all_fpaths) == 0:
            print('no files found with pattern: ', fpath_pattern)
            continue
        print(f'found {len(all_fpaths)} files')

        for fpath in all_fpaths:
            data = torch.load(fpath)
            metrics_test = data[3]
            v_test = metrics_test[m2i[metric]]
            if metric == 'ROCAUC':
                v_test = -v_test
            res_test_seeds.append(v_test)
            #
            metrics_valid = data[1][-1]
            v_valid = metrics_valid[m2i[metric]]
            if metric == 'ROCAUC':
                v_valid = -v_valid
            res_val_seeds.append(v_valid)
            #
            mem = int(np.ceil(data[4] / 1024))
            memories_seeds.append(mem)
            tt = np.mean(data[5]['train.forward'])
            timecosts_seeds.append(tt)

        testset[model_name] = res_test_seeds
        valset[model_name] = res_val_seeds
        memories[model_name] = memories_seeds
        timecosts[model_name] = timecosts_seeds

    return valset, testset, memories, timecosts

if __name__ == '__main__':
    
    # python agg_results.py --dataset <> --task cls --framework induc --metric ROCAUC --dynedge none --lr 0.001 --seed '*' --exp-name ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Brain10')
    parser.add_argument('--task', type=str, default='cls', help='cls/reg')
    parser.add_argument('--framework', type=str, default='trans', help="trans/induc")
    parser.add_argument('--metric', type=str, default='ROCAUC', help="ROCAUC/CE/ERR")
    parser.add_argument('--dynedge', type=str, default=None, help="dense/none/`None`")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-wd', '--weight-decay', type=str, default='1.0e-5', help="weigth decay")
    parser.add_argument('--patience', type=int, default=-1, help="patience for early stopping, -1 for no early stopping")
    parser.add_argument('--seed', type=str, default="*", help="* for average all seeds, or a value 56/57/...")
    parser.add_argument('--exp-name', type=str, default="", help="experiment name")
    parser.add_argument('--model_plot', type=str, default=None, help="name of model to plot, EvoGCNOx2, EvoGCNHx2, GCNx2oGRU, DySATx2, GCRNM2x2, DCRNNx2, TGATx2, TGNOptimLx2, GRUoGCN2x2, IDGNN")
    parser.add_argument('--best-eval', action='store_true', help="plot best eval")
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    framework  = args.framework
    metric = args.metric
    model_names = ['EvoGCNOx2', 'EvoGCNHx2', 'GCNx2oGRU', 'DySATx2', 
                   'GCRNM2x2', 'DCRNNx2', 'TGATx2', 'TGNOptimLx2', 
                   'GRUoGCN2x2', 'IDGNN']
    dynedge = args.dynedge
    lr = args.lr
    patience = args.patience
    seed = eval(args.seed) if args.seed != '*' else '*'
    exp_name = args.exp_name
    wd = cast(str, args.weight_decay)
    besteval = args.best_eval

    if args.model_plot is not None:
        plot_curves(dataset, framework, metric, task, args.model_plot, dynedge=dynedge, lr=lr, wd=wd, patience=patience, seed=seed, exp_name=exp_name)
    else:
        valset, testset, memories, timecosts = parse_logs(dataset, framework, metric, task, model_names, 
                                                          dynedge=dynedge, lr=lr, wd=wd, patience=patience, 
                                                          seed=seed, exp_name=exp_name, besteval=besteval)
        print(f"{metric} of {framework} task on {dataset}")
        for method in model_names:
            print(f'{method}')

        print('Validation')
        for method in model_names:
            if method not in valset:
                mean, std = None, None
                print(f'{mean}\u00B1{std}')
            else:
                mean = np.mean(valset[method])
                std = np.std(valset[method])

                print(f'{mean * 100:.4f}\u00B1{std * 100:.4f}')
        
        print('Test')
        for method in model_names:
            if method not in testset:
                mean, std = None, None
                print(f'{mean}\u00B1{std}')
            else:
                mean = np.mean(testset[method])
                std = np.std(testset[method])

                print(f'{mean * 100:.4f}\u00B1{std * 100:.4f}')


        print('Memory')
        for method in model_names:
            if method not in memories:
                mean, std = None, None
                print(f'{mean}\u00B1{std}')
            else:
                mean = np.mean(memories[method])
                std = np.std(memories[method])

                print(f'{mean:.4f}\u00B1{std:.4f}')

        print('Time')
        for method in model_names:
            if method not in timecosts:
                mean, std = None, None
                print(f'{mean}\u00B1{std}')
            else:
                mean = np.mean(timecosts[method])
                std = np.std(timecosts[method])

                print(f'{mean:.4f}\u00B1{std:.4f}')

