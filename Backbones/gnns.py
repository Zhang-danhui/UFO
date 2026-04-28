import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.utils import expand_as_pair


gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        dims = [args.d_data] + args.GCN_args["h_dims"]
        self.dropout = args.GCN_args["dropout"]
        self.h_dim = args.GCN_args["h_dims"][-1]

        self.gcn_layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.each_classes = args.n_cls_per_task
        self.num_tasks = getattr(args, "n_tasks", None)
        self.n_cls = getattr(args, "n_cls", None)
        if self.n_cls is None:
            if self.num_tasks is None:
                raise ValueError("GCN requires args.n_cls or args.n_tasks for single-head Task-IL.")
            self.n_cls = self.num_tasks * self.each_classes
        self.classifier = nn.Linear(self.h_dim, self.n_cls)
        self.second_last_h = None

    def reset_parameters(self):
        for layer in self.gcn_layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward_to_xa(self, g, features):
        h = features
        for layer in self.gcn_layers:
            h, _ = layer(g, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        self.second_last_h = h
        return h

    def forward_to_xa_batch(self, blocks, features):
        h = features
        for i, layer in enumerate(self.gcn_layers):
            h, _ = layer.forward_batch(blocks[i], h)
            h = h[: blocks[i].num_dst_nodes()]
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        self.second_last_h = h
        return h

    def forward_batch(self, blocks, features, task=None):
        h = self.forward_to_xa_batch(blocks, features)
        return self.forward_from_xa(h, task)

    def forward_from_xa(self, xa, task=None):
        logits = self.classifier(xa)
        if task is None:
            return logits
        offset1 = task * self.each_classes
        offset2 = offset1 + self.each_classes
        return logits[:, offset1:offset2]


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, negative_slope=0.2):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, feat):
        elist = []
        graph = graph.local_var().to(f"cuda:{feat.get_device()}")
        h = self.linear(feat)
        h = h * graph.ndata["out_norm"]
        graph.ndata["h"] = h
        graph.update_all(gcn_msg, gcn_reduce)
        h = graph.ndata["h"] * graph.ndata["in_norm"]

        graph.apply_edges(
            lambda edges: {"e": th.sum(th.mul(edges.src["h"], th.tanh(edges.dst["h"])), 1)}
        )
        e = self.leaky_relu(graph.edata.pop("e"))
        elist.append(edge_softmax(graph, e))
        return h, elist

    def forward_batch(self, block, feat):
        elist = []
        block = block.local_var().to(f"cuda:{feat.get_device()}")
        feat_src, _ = expand_as_pair(feat)

        x = feat_src * block.srcdata["out_norm"]
        block.srcdata["x"] = x
        block.update_all(fn.copy_u("x", "m"), fn.sum("m", "x_agg"))
        x_agg = block.dstdata["x_agg"] * block.dstdata["in_norm"]
        h = self.linear(x_agg)
        return h, elist

    def reset_parameters(self):
        gain = nn.init.calculate_gain("leaky_relu", 0.2)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

