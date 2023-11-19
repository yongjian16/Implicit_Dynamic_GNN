R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import torch
import os
from typing import Optional, List, Dict, Tuple, cast
from ..meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge
import collections
import pickle
from ..model.utils import get_spectral_rad, aug_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from scipy.sparse import coo_array, coo_matrix
import benchtemp as bt
import networkx as nx


class DynamicClassification(object):
    R"""
    Dynamic node classification over the whole window dataset.
    """
    #
    SOURCE: str

    def __init__(
        self,
        dirname: str,
        /,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        self.from_raw(dirname)
        self.sanitize_edge()

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        matrices = (
            onp.load(os.path.join(dirname, "{:s}.npz".format(self.SOURCE)))
        )
        feats = matrices["attmats"] # Brain10 (5000, 12, 20)
        adjmats = matrices["adjs"] # Brain10 (12, 5000, 5000)
        onehots = matrices["labels"] # Brain10 (5000, 10)
        #
        # A_list = [ sparse_mx_to_torch_sparse_tensor(aug_normalized_adjacency(coo_matrix(adj, dtype=float))) for adj in adjmats]
        # # A_list = [ torch.tensor(adj, dtype=torch.float).to(device).to_sparse() for adj in data['adjs']]
        # A_rho = [get_spectral_rad(A) for A in A_list]
        # self.A_rho = A_rho
        # self.A_list = A_list
        #
        # Safety check.
        if onp.any(adjmats < 0):
            # UNEXPECT:
            # Edge weights must be non-negative.
            raise NotImplementedError("Get invalid negative edge weights.")
        if onp.any(onp.sum(onehots > 0, axis=1) != 1):
            # UNEXPECT:
            # Node label must be unique.
            raise NotImplementedError("Get empty or duplicate node labels.")

        #
        self.raw_node_feats = feats
        (_, self.raw_node_labels) = onp.nonzero(onehots)
        (_, self.num_labels) = onehots.shape
        self.label_counts = onp.sum(onehots, axis=0).tolist()

        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        (_, self.num_times, _) = feats.shape
        for t in range(self.num_times):
            #
            (dsts, srcs) = onp.nonzero(adjmats[t]) # Brain10 dsts.shape = srcs.shape = [(154094,), (164190,), ...]
            weights = adjmats[t, dsts, srcs].astype(float) # Brain10 weights.shape = (154094,)
            if onp.any(weights != 1):
                # UNEXPECT:
                # Node label must be unique.
                raise NotImplementedError("Edge has non-0/1 weight.")
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
            
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)

    def sanitize_edge(self, /) -> None:
        R"""
        Santiize edge data.
        """
        # All dynamic classification tasks are assumed to be directed graphs.
        self.edge_srcs = self.raw_edge_srcs
        self.edge_dsts = self.raw_edge_dsts
        self.edge_feats = self.raw_edge_feats
        self.edge_hetero = False
        self.edge_symmetric = False

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1))
        num_nodes = len(node_feats)
        node_labels = onp.reshape(self.raw_node_labels, (num_nodes, 1))

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
            )
        )
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=False)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        # customized info
        if hasattr(self, 'lab_avail_nodes'):
            metaset.lab_avail_nodes = self.lab_avail_nodes
        else:
            metaset.lab_avail_nodes = None

        if hasattr(self, 'A_rho'):
            metaset.A_rho = self.A_rho
            metaset.A_list = self.A_list
        return metaset


class Reddit4(DynamicClassification):
    #
    SOURCE = "Reddit4"


class DBLP5(DynamicClassification):
    #
    SOURCE = "DBLP5"


class Brain10(DynamicClassification):
    #
    SOURCE = "Brain10"


class DynCSL(DynamicClassification):
    #
    SOURCE = "DynCSL"

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        (data, properties) = (
            torch.load(os.path.join(dirname, "tgnn-power-v2.pt"))
        )
        num_nodes = properties["num_nodes"]
        num_labels = properties["num_labels"]
        num_times = properties["num_timestamps"]
        print(properties)
        
        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        graph_label_buf = []
        for (l, (graph_pair, label)) in enumerate(data):
            #
            (feats, (adjlists, _)) = graph_pair
            for adjlist in adjlists:
                #
                src_snap = []
                dst_snap = []
                for (src, dsts) in enumerate(adjlist):
                    #
                    for dst in dsts:
                        #
                        src_snap.append(src)
                        dst_snap.append(dst)
                if torch.min(feats) != 1 or torch.max(feats) != 1:
                    # UNEXPECT:
                    # DynCSL node features are non-trivial.
                    raise NotImplementedError(
                        "DynCSL node features are non-trivial.",
                    )
                if len(src_snap) != num_nodes * 5:
                    # UNEXPECT:
                    # DynCSL is incomplete.
                    raise NotImplementedError(
                        "Incomplete DynCSL source nodes.",
                    )
                if len(dst_snap) != num_nodes * 5:
                    # UNEXPECT:
                    # DynCSL is incomplete.
                    raise NotImplementedError(
                        "Incomplete DynCSL destination nodes.",
                    )
                self.raw_edge_srcs.append(onp.array(src_snap))
                self.raw_edge_dsts.append(onp.array(dst_snap))
                self.raw_edge_feats.append(onp.ones((num_nodes * 5,)))
                graph_label_buf.append(-1)
            graph_label_buf[-1] = label.item()

            if graph_label_buf[-1] != l % num_labels:
                # UNEXPECT:
                # DynCSL is incomplete.
                raise NotImplementedError(
                    "DynCSL labels should averagely distribute in raw "
                    "sequence.",
                )
        graph_labels = onp.array(graph_label_buf)
        if len(graph_labels) != num_times * len(data):
            # UNEXPECT:
            # DynCSL is incomplete.
            raise NotImplementedError(
                "Some DynCSL has not have {:d} timestamps.".format(num_times),
            )

        #
        self.raw_node_feats = onp.array([[[0.0, 1.0]]])
        self.raw_node_feats = (
            onp.tile(self.raw_node_feats, (num_nodes, len(graph_labels), 1))
        )
        self.raw_node_labels = (
            onp.reshape(graph_labels, (1, len(graph_label_buf), 1))
        )
        self.raw_node_labels = (
            onp.tile(self.raw_node_labels, (num_nodes, 1, 1))
        )

        #
        self.win_size = num_times
        self.num_times = self.win_size * len(data)
        self.num_labels = num_labels
        self.timestamps = onp.arange(self.win_size, dtype=onp.float64)
        self.timestamps = onp.tile(self.timestamps, (len(data),))

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1))
        (num_nodes, _, num_times) = node_feats.shape
        node_labels = (
            onp.reshape(self.raw_node_labels, (num_nodes, 1, num_times))
        )

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
            )
        )
        metaset.set_win_shift(self.win_size)
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=True)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        return metaset
    

class IMDB(DynamicClassification):
    #
    SOURCE = "IMDB"

    def get_general_info(self, dir):
        # step 0: get basic info
        with open(dir + "/graph.info", "r") as f:
            num_tws = int(f.readline())
            num_nodes = int(f.readline())
            num_attrs = int(f.readline())
        return num_tws, num_nodes, num_attrs

    def get_labels(self, dir, num_nodes, default_lab_val=-1):
        with open(dir + "/graph.lab", "r") as f:
            l_labeled_nd = []
            y = onp.zeros([num_nodes], dtype=onp.int64)
            for line in f.readlines():  # assign labels
                token = line.split("::")
                nd = int(token[0])
                lb = int(token[1])
                l_labeled_nd.append(nd)
                y[nd] = lb

            for nd in range(num_nodes):  # put indicators for nodes who do not have labels
                if nd not in l_labeled_nd:
                    y[nd] = default_lab_val
        return y, l_labeled_nd
    
    def get_adjmat_from_file(self, dir, num_nodes, num_times, nodeval2indices=None):
        all_time_vals = set()
        edgeinfo_avail_nodes = set()
        with open(dir + "/graph.tedgelist", "r") as f:
            for line in f.readlines():
                token = line.split("::")
                t = int(token[2])
                s = int(token[0])
                d = int(token[1])
                all_time_vals.add(t)
                edgeinfo_avail_nodes.add(s)
                edgeinfo_avail_nodes.add(d)
        all_time_vals = sorted(all_time_vals)
        # all_node_vals = sorted(all_node_vals)
        timeval2indices = {t: idx for idx, t in enumerate(all_time_vals)}
        # nodeval2indices = {n: idx for idx, n in enumerate(all_node_vals)}
        assert len(timeval2indices) == num_times, 'Inconsistencies found!'
        # assert len(nodeval2indices) == num_nodes, 'Inconsistancies found!' # no need for nodes

        adjmats = onp.zeros((num_times, num_nodes, num_nodes))

        with open(dir + "/graph.tedgelist", "r") as f:
            for line in f.readlines():
                token = line.split("::")
                t = int(token[2])
                s = int(token[0])
                d = int(token[1])
                t = timeval2indices[t]
                # s = nodeval2indices[s]
                # d = nodeval2indices[d]
                adjmats[t, s, d] = 1

        return adjmats, timeval2indices, edgeinfo_avail_nodes

    def get_node_static_attrs(self, dir, num_nodes, num_node_feat):
        with open(dir + "/graph.attr", "r") as f:
            l_attr_nd = set()
            x = onp.zeros([num_nodes, num_node_feat])
            for line in f.readlines():  # assign labels
                token = line.split("::")
                nd = int(token[0])
                attr = token[1:]
                attr = [int(float(a)) for a in attr]
                x[nd] = attr
                l_attr_nd.add(nd)

            for nd in range(num_nodes):  # put indicators for nodes who do not have labels
                if nd not in l_attr_nd:
                    x[nd] = [-1] * num_node_feat
        return x, l_attr_nd
    
    
    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        num_times, num_nodes, num_node_attrs = self.get_general_info(dirname)
        num_labels = 2
        
        #
        feats, feat_avail_nodes = self.get_node_static_attrs(dirname, num_nodes, num_node_attrs) # (num_nodes, num_feats)
        print(f'Missing feat nodes: {sum(onp.alltrue(feats == -1, axis=1))}')

        feats = torch.tensor(feats).unsqueeze(1).repeat(1, num_times, 1).numpy()
        adjmats, self.timeval2indices, edgeinfo_avail_nodes = \
            self.get_adjmat_from_file(dirname, num_nodes, num_times)
        
        self.raw_node_feats = feats
        self.raw_node_labels, self.lab_avail_nodes = self.get_labels(dirname, num_nodes, default_lab_val=-1)
        assert len(set(self.lab_avail_nodes).intersection(set(feat_avail_nodes))) == len(feat_avail_nodes)
        miss_lb_nodes = onp.where(self.raw_node_labels == -1)[0]
        print(f'Missing-label nodes: {len(miss_lb_nodes)}/{num_nodes}')
        avail_node_missing_lb = set(miss_lb_nodes).intersection(set(edgeinfo_avail_nodes))
        print(f'Edgeinfo avail nodes but missing label: {len(avail_node_missing_lb)}')

        self.num_labels = num_labels
        self.label_counts = [sum(self.raw_node_labels == i) for i in range(self.num_labels)]
        lb_cnts = onp.array(self.label_counts)
        print(f'label counts: {self.label_counts} ({[f"{v:.2f}%" for v in lb_cnts / lb_cnts.sum()]})')
        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        (_, self.num_times, _) = feats.shape

        for t in range(self.num_times):
            #
            (dsts, srcs) = onp.nonzero(adjmats[t])
            weights = adjmats[t, dsts, srcs].astype(float)
            if onp.any(weights != 1):
                # UNEXPECT:
                # Node label must be unique.
                raise NotImplementedError("Edge has non-0/1 weight.")
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)


class UIHC(DynamicClassification):
    #
    SOURCE = "UIHC"

    def from_raw(self, dirname: str, /, prevalance_label=True) -> None:
        R"""
        Load from raw data.
        """
        #
        if prevalance_label: 
            fname = "UIHC_30days_prevalence.pickle"
        else:
            fname = "UIHC_30days_incidence.pickle"

        with open(os.path.join(dirname, fname), 'rb') as f:
            matrices = pickle.load(f)

        feats = matrices["attmats"] # (num_times, num_nodes, num_feats)
        feats = onp.transpose(feats, (1, 0, 2)) # (num_nodes, num_times, num_feats)

        adjmats = matrices["adjs"] 
        onehots = matrices["labels"] 

        # Safety check.
        if onp.any(adjmats < 0):
            # UNEXPECT:
            # Edge weights must be non-negative.
            raise NotImplementedError("Get invalid negative edge weights.")
        if onp.any(onp.sum(onehots > 0, axis=-1) != 1):
            # UNEXPECT:
            # Node label must be unique.
            raise NotImplementedError("Get empty or duplicate node labels.")

        #
        self.raw_node_feats = feats
        (_, _, self.raw_node_labels) = onp.nonzero(onehots)
        (_, _, self.num_labels) = onehots.shape
        self.label_counts = onp.sum(onehots, axis=0).sum(axis=0).tolist()

        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        (_, self.num_times, _) = feats.shape
        for t in range(self.num_times):
            #
            (dsts, srcs) = onp.nonzero(adjmats[t]) # Brain10 dsts.shape = srcs.shape = [(154094,), (164190,), ...]
            weights = adjmats[t, dsts, srcs].astype(float) # Brain10 weights.shape = (154094,)
            # if onp.any(weights != 1):
            #     raise NotImplementedError("Edge has non-0/1 weight.")
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)
    
    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1))
        (num_nodes, _, num_times) = node_feats.shape
        node_labels = (
            onp.reshape(self.raw_node_labels, (num_nodes, 1, num_times))
        )

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
            )
        )
        # metaset.set_win_shift(1, data_source=self.SOURCE) # UIHC is a daily dataset
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=True)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        # customized info
        if hasattr(self, 'lab_avail_nodes'):
            metaset.lab_avail_nodes = self.lab_avail_nodes
        else:
            metaset.lab_avail_nodes = None

        return metaset
    


class TgbnGenre(DynamicClassification):

    SOURCE = "tgbn-genre"

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        # num_times, num_nodes, num_node_attrs = self.get_general_info(dirname)
        # num_labels = 2

        # For example, if you are training , you should create a training  RandEdgeSampler based on the training dataset.
        data = bt.nc.DataLoader(dataset_path="./src/benchtemp_datasets/", dataset_name='mooc')

        # dataloader for dynamic node  classification task

        full_data, node_features, edge_features, train_data, val_data, test_data = data.load()
        import pdb;pdb.set_trace()
        
        #
        self.raw_node_feats = node_features # static
        self.raw_node_labels = full_data.labels # 1: 4066, 0:407683
        
        self.num_labels = len(set(self.raw_node_labels))
        self.label_counts = [sum(self.raw_node_labels == i) for i in range(self.num_labels)]
        lb_cnts = onp.array(self.label_counts)
        print(f'label counts: {self.label_counts} ({[f"{v:.2f}%" for v in lb_cnts / lb_cnts.sum()]})')
        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        (_, self.num_times, _) = feats.shape

        for t in range(self.num_times):
            #
            (dsts, srcs) = onp.nonzero(adjmats[t])
            weights = adjmats[t, dsts, srcs].astype(float)
            if onp.any(weights != 1):
                # UNEXPECT:
                # Node label must be unique.
                raise NotImplementedError("Edge has non-0/1 weight.")
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)


class MOOC(DynamicClassification):

    SOURCE = "mooc"

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        # num_times, num_nodes, num_node_attrs = self.get_general_info(dirname)
        # num_labels = 2

        # For example, if you are training , you should create a training  RandEdgeSampler based on the training dataset.
        data = bt.nc.DataLoader(dataset_path="./src/benchtemp_datasets/", dataset_name='wikipedia')

        # dataloader for dynamic node  classification task

        full_data, node_features, edge_features, train_data, val_data, test_data = data.load()        
        #
        self.raw_node_feats = node_features # static
        self.raw_edge_labels = full_data.labels # 1: 4066, 0:407683
        
        # self.num_labels = len(set(self.raw_node_labels))
        # self.label_counts = [sum(self.raw_node_labels == i) for i in range(self.num_labels)]
        # lb_cnts = onp.array(self.label_counts)
        # print(f'label counts: {self.label_counts} ({[f"{v:.2f}%" for v in lb_cnts / lb_cnts.sum()]})')
        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        self.raw_node_labels = [] 
    
        self.num_times = 200 # TODO

        start_time = min(full_data.timestamps)
        end_time = max(full_data.timestamps)
        bin_edges = onp.linspace(start_time, end_time, self.num_times+1, endpoint=True)

        import pdb;pdb.set_trace()

        for t in range(self.num_times):
            snapshot_start = bin_edges[t]
            snapshot_end = bin_edges[t + 1]
            snapshot_mask = (full_data.timestamps >= snapshot_start) & (full_data.timestamps <= snapshot_end)
            #
            snapshot_src = full_data.sources[snapshot_mask]
            snapshot_dst = full_data.destinations[snapshot_mask]
            snapshot_edge_feats = edge_features[snapshot_mask]
            self.raw_edge_srcs.append(snapshot_src)
            self.raw_edge_dsts.append(snapshot_dst)
            self.raw_edge_feats.append(snapshot_edge_feats)

        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1))
        num_nodes = len(node_feats)
        node_labels = onp.reshape(self.raw_node_labels, (num_nodes, 1))

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
            )
        )
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=False)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        # customized info
        if hasattr(self, 'lab_avail_nodes'):
            metaset.lab_avail_nodes = self.lab_avail_nodes
        else:
            metaset.lab_avail_nodes = None

        if hasattr(self, 'A_rho'):
            metaset.A_rho = self.A_rho
            metaset.A_list = self.A_list
        return metaset
    

from check_data import extract_info

class SFHH(DynamicClassification):

    SOURCE = "SFHH"

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        # num_times, num_nodes, num_node_attrs = self.get_general_info(dirname)
        # num_labels = 2

        # For example, if you are training , you should create a training  RandEdgeSampler based on the training dataset.
        node_idx_mapping, graphs, label_matrix = extract_info(self.SOURCE)
        # create one-hot encoding for node_features
        node_features = onp.zeros((len(node_idx_mapping), len(node_idx_mapping)))
        for i in range(len(node_idx_mapping)):
            node_features[i, i] = 1
        # add all the nodes that are not in the graph
        for graph in graphs:
            graph.add_nodes_from(list(set(node_idx_mapping.values()) - set(graph.nodes())))

        self.num_times = len(graphs)
        self.num_nodes = len(node_idx_mapping)
        self.raw_node_feats = node_features # static
        # make unkwown labels -1 to 0 as susceptible
        label_matrix[label_matrix == -1] = 0
        label_matrix = label_matrix.astype(int)[..., None]
        assert label_matrix.shape == (self.num_times, self.num_nodes, 1)
        self.raw_node_labels = label_matrix # num_timesteps, num_nodes {0, 1, 2}
        
        self.num_labels = 3
        # self.label_counts = [sum(self.raw_node_labels == i) for i in range(self.num_labels)]
        # lb_cnts = onp.array(self.label_counts)
        # print(f'label counts: {self.label_counts} ({[f"{v:.2f}%" for v in lb_cnts / lb_cnts.sum()]})')
        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
    
        for t in range(self.num_times):
            graph = graphs[t]
            adj_mat = nx.adjacency_matrix(graph).todense()
            #
            (dsts, srcs) = onp.nonzero(adj_mat)
            weights = adj_mat[dsts, srcs].astype(float)
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
            
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        # node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1))
        # num_nodes = len(node_feats)
        node_feats = self.raw_node_feats[..., None].repeat(self.num_times, axis=-1)
        node_labels = onp.transpose(self.raw_node_labels, (1, 2, 0)) # (num_nodes, num_feats, num_times)

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
            )
        )
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=True)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        # customized info
        if hasattr(self, 'lab_avail_nodes'):
            metaset.lab_avail_nodes = self.lab_avail_nodes
        else:
            metaset.lab_avail_nodes = None

        if hasattr(self, 'A_rho'):
            metaset.A_rho = self.A_rho
            metaset.A_list = self.A_list

        return metaset