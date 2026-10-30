import random
import pickle
import numpy as np
import torch
from torch import Tensor, device, dtype
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.data import (
    AmazonCoBuyPhotoDataset,
    CoauthorCSDataset,
    CoraGraphDataset,
    CoraFullDataset,
    RedditDataset,
    WikiCSDataset,
    register_data_args,
)
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
import copy
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from torch_geometric.datasets import AttributedGraphDataset
except ImportError:
    AttributedGraphDataset = None

class Linear_IL(nn.Linear):
    def forward(self, input: Tensor, n_cls=10000, normalize = True) -> Tensor:
        if normalize:
            return F.linear(F.normalize(input,dim=-1), F.normalize(self.weight[0:n_cls],dim=-1), bias=None)
        else:
            return F.linear(input, self.weight[0:n_cls], bias=None)

def accuracy(logits, labels, cls_balance=True, ids_per_cls=None):
    _, pred = torch.max(logits, dim=1)

    if not cls_balance:
        return (pred == labels).float().mean().item()

    assert ids_per_cls is not None, "ids_per_cls is required when cls_balance=True"
    accs = []
    for ids in ids_per_cls:
        if len(ids) == 0:
            continue
        ids_t = torch.as_tensor(ids, device=labels.device, dtype=torch.long)
        accs.append((pred[ids_t] == labels[ids_t]).float().mean())
    return torch.stack(accs).mean().item() if len(accs) > 0 else 0.0


def mean_AP(args,logits, labels, cls_balance=True, ids_per_cls=None):
    eval_ogb = Evaluator(args.dataset)
    pos = (F.sigmoid(logits)>0.5)
    APs = 0
    if cls_balance:
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        input_dict = {"y_true": labels, "y_pred": logits}

        eval_result_ogb = eval_ogb.eval(input_dict)
        for c,ids in enumerate(ids_per_cls):
            TP_ = (pos[ids,c]*labels[ids,c]).sum()
            FP_ = (pos[ids,c]*(labels[ids, c]==False)).sum()
            med0 = TP_ + FP_ + 0.0001
            med1 = TP_ / med0
            APs += med1
        med2 = APs/labels.shape[1]

                                                             
                                            

        return med2.item()
import torch
import numpy as np
import dgl

                              
                                                  

def evaluate_batch(args, model, g, features, labels, mask,
                   label_offset1, label_offset2,
                   cls_balance=True, ids_per_cls=None,
                   debug=True):

    model.eval()
    device = f'cuda:{args.gpu}'

                                           
    task = label_offset1 // args.n_cls_per_task

            
    C_local = label_offset2 - label_offset1

    with torch.no_grad():
                                                       
        mask_ids_cpu = torch.as_tensor(mask, dtype=torch.long)                     
        mask_ids_gpu = mask_ids_cpu.to(device)

                                           
                                               
        eval_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

        dataloader = dgl.dataloading.DataLoader(
            g.cpu(),
            mask_ids_cpu.tolist(),                       
            eval_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        all_logits = []
        all_y = []

        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            blocks = [b.to(device=device) for b in blocks]

                                                           
            x = blocks[0].srcdata['feat']                                  
            y_global = blocks[-1].dstdata['label'].view(-1)                          
            y_local = (y_global - label_offset1).long()                                          

                                   
            logits = model.forward_batch(blocks, x, task=task)
            if isinstance(logits, tuple):
                logits = logits[0]

            n_dst = blocks[-1].num_dst_nodes()
            logits = logits[:n_dst]

            if logits.size(1) == C_local:
                pass
            elif logits.size(1) >= label_offset2:
                logits = logits[:, label_offset1:label_offset2]
            else:
                raise RuntimeError(
                    f"[evaluate_batch] logits shape={tuple(logits.shape)} "
                    f"cannot match C_local={C_local} or slice [{label_offset1}:{label_offset2}]"
                )

            all_logits.append(logits)
            all_y.append(y_local)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

             
        logits_eval = torch.cat(all_logits, dim=0)
        y_eval = torch.cat(all_y, dim=0)

                                             
        pred = logits_eval.argmax(dim=1)
        micro = (pred == y_eval).float().mean().item()
        return micro

def evaluate(model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        output, _ = model(g, features)
        logits = output[:, label_offset1:label_offset2]
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

class incremental_graph_trans_(nn.Module):
    def __init__(self, dataset, n_cls):
        super().__init__()
                              
        self.graph, self.labels = dataset[0]
                                                       
                                                   
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_cls = n_cls
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_nodes = self.labels.shape[0]
        self.tr_va_te_split = dataset[1]

    def get_graph(self, tasks_to_retain=[], node_ids = None, remove_edges=True):
                               
                                                                
                                                                 
        node_ids_ = copy.deepcopy(node_ids)
        node_ids_retained = []
        ids_train_old, ids_valid_old, ids_test_old = [], [], []
        if len(tasks_to_retain) > 0:
                                               
            for t in tasks_to_retain:
                ids_train_old.extend(self.tr_va_te_split[t][0])
                ids_valid_old.extend(self.tr_va_te_split[t][1])
                ids_test_old.extend(self.tr_va_te_split[t][2])
                node_ids_retained.extend(self.tr_va_te_split[t][0] + self.tr_va_te_split[t][1] + self.tr_va_te_split[t][2])
            subgraph_0 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True)
            if node_ids_ is None:
                subgraph = subgraph_0
        if node_ids_ is not None:
                                     
            if not isinstance(node_ids_[0],list):
                                                               
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_, store_ids=True)
                if remove_edges:
                                                                                  
                    n_edges = subgraph_1.edges()[0].shape[0]
                    subgraph_1.remove_edges(list(range(n_edges)))
            elif isinstance(node_ids_[0],list):
                                                           
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_[0], store_ids=True)                                                       
                node_ids_.pop(0)
                for ids in node_ids_:
                                               
                    subgraph_1 = dgl.batch([subgraph_1,dgl.node_subgraph(self.graph, ids, store_ids=True)])

            if len(tasks_to_retain)==0:
                subgraph = subgraph_1

        if len(tasks_to_retain)>0 and node_ids is not None:
            subgraph = dgl.batch([subgraph_0,subgraph_1])

        old_ids = subgraph.ndata['_ID'].cpu()
        ids_train = [(old_ids == i).nonzero()[0][0].item() for i in ids_train_old]
        ids_val = [(old_ids == i).nonzero()[0][0].item() for i in ids_valid_old]
        ids_test = [(old_ids == i).nonzero()[0][0].item() for i in ids_test_old]
        node_ids_per_task_reordered = []
        for c in tasks_to_retain:
            ids = (subgraph.ndata['label'] == c).nonzero()[:, 0].view(-1).tolist()
            node_ids_per_task_reordered.append(ids)
        subgraph = dgl.add_self_loop(subgraph)
        deg_in  = subgraph.in_degrees().float().clamp(min=1)
        deg_out = subgraph.out_degrees().float().clamp(min=1)
        subgraph.ndata['in_norm']  = deg_in.pow(-0.5).unsqueeze(1)          
        subgraph.ndata['out_norm'] = deg_out.pow(-0.5).unsqueeze(1)         
        
        return subgraph, node_ids_per_task_reordered, [ids_train, ids_val, ids_test]

def train_valid_test_split(ids,ratio_valid_test):
    va_te_ratio = sum(ratio_valid_test)
    train_ids, va_te_ids = train_test_split(ids, test_size=va_te_ratio)
    return [train_ids] + train_test_split(va_te_ids, test_size=ratio_valid_test[1]/va_te_ratio)


class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name='ogbn-arxiv',IL='class',default_split=False,ratio_valid_test=None,args=None):
        r""""
        name: name of the dataset
        IL: use task- or class-incremental setting
        default_split: if True, each class is split according to the splitting of the original dataset, which may cause the train-val-test ratio of different classes greatly different
        ratio_valid_test: in form of [r_val,r_test] ratio of validation and test set, train set ratio is directly calculated by 1-r_val-r_test
        """

                                                                                             
        if name[0:4] == 'ogbn':
            data = DglNodePropPredDataset(name, root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name in ['CoraFullDataset', 'CoraFull','corafull', 'CoraFull-CL','Corafull-CL']:
            data = CoraFullDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['Photo', 'photo', 'Photo-CL', 'AmazonPhoto', 'AmazonPhoto-CL']:
            data = AmazonCoBuyPhotoDataset()
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)
        elif name in ['CS', 'cs', 'CS-CL', 'CoauthorCS', 'CoauthorCS-CL']:
            data = CoauthorCSDataset()
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)
        elif name in ['WikiCS', 'wikics', 'WikiCS-CL']:
            data = WikiCSDataset()
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)
        elif name in ['BlogCatalog', 'blogcatalog', 'BlogCatalog-CL', 'blogcatalog-CL']:
            if AttributedGraphDataset is None:
                raise ImportError('torch_geometric is required to load BlogCatalog')

            data = AttributedGraphDataset(root=args.ori_data_path, name='blogcatalog')
            pyg_data = data[0]
            graph = dgl.graph(
                (pyg_data.edge_index[0], pyg_data.edge_index[1]),
                num_nodes=pyg_data.num_nodes
            )
            graph.ndata['feat'] = pyg_data.x.float()

            if pyg_data.y.dim() == 1:
                label = pyg_data.y.view(-1, 1).long()
            elif pyg_data.y.dim() == 2 and pyg_data.y.shape[1] == 1:
                label = pyg_data.y.long()
            elif pyg_data.y.dim() == 2:
                row_sum = pyg_data.y.sum(dim=1)
                if not torch.all(row_sum == 1):
                    raise ValueError(
                        'BlogCatalog labels are multi-label in the current data source, '
                        'but this pipeline expects a single-label 39-class version.'
                    )
                label = pyg_data.y.argmax(dim=1, keepdim=True).long()
            else:
                raise ValueError(f'Unsupported BlogCatalog label shape: {tuple(pyg_data.y.shape)}')
        elif name in ['Facebook', 'facebook', 'Facebook-CL', 'facebook-CL']:
            if AttributedGraphDataset is None:
                raise ImportError('torch_geometric is required to load Facebook')

            data = AttributedGraphDataset(root=args.ori_data_path, name='Facebook')
            pyg_data = data[0]
            graph = dgl.graph(
                (pyg_data.edge_index[0], pyg_data.edge_index[1]),
                num_nodes=pyg_data.num_nodes
            )
            graph.ndata['feat'] = pyg_data.x.float()

            if pyg_data.y.dim() == 1:
                label = pyg_data.y.view(-1, 1).long()
            elif pyg_data.y.dim() == 2 and pyg_data.y.shape[1] == 1:
                label = pyg_data.y.long()
            elif pyg_data.y.dim() == 2:
                row_sum = pyg_data.y.sum(dim=1)
                if not torch.all(row_sum == 1):
                    raise ValueError(
                        'Facebook labels are multi-label in the current data source, '
                        'but this pipeline expects a single-label version.'
                    )
                label = pyg_data.y.argmax(dim=1, keepdim=True).long()
            else:
                raise ValueError(f'Unsupported Facebook label shape: {tuple(pyg_data.y.shape)}')
        elif name in ['reddit','Reddit','Reddit-CL']:
            data = RedditDataset(self_loop=False)
            graph, label = data[0], data[0].ndata['label'].view(-1, 1)
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name == 'Products-CL':
            data = DglNodePropPredDataset('ogbn-products', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        else:
            print('invalid data name')
        n_cls = data.num_classes
        cls = [i for i in range(n_cls)]
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map}
        for c in cls_sizes:
            if cls_sizes[c] < 2:
                cls.remove(c)                                                                                             
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        n_cls = len(cls)
        if default_split:
            split_idx = data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
                "test"].tolist()
            tr_va_te_split = {c: [list(set(cls_id_map[c]).intersection(set(train_idx))),
                                  list(set(cls_id_map[c]).intersection(set(valid_idx))),
                                  list(set(cls_id_map[c]).intersection(set(test_idx)))] for c in cls}

        elif not default_split:
            split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            try:
                tr_va_te_split = pickle.load(open(split_name, 'rb'))                                                                    
            except:
                if ratio_valid_test[1] > 0:
                    tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
                                      for c in
                                      cls}
                    print(f'splitting is {ratio_valid_test}')
                elif ratio_valid_test[1] == 0:
                    tr_va_te_split = {c: [cls_id_map[c], [], []] for c in
                                      cls}
                with open(split_name, 'wb') as f:
                    pickle.dump(tr_va_te_split, f)
            else:
                # Match first-time split RNG consumption once per experiment.
                if ratio_valid_test[1] > 0 and not getattr(args, '_split_rng_consumed', False):
                    for c in cls:
                        train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
            args._split_rng_consumed = True
        super().__init__([[graph, label], tr_va_te_split], n_cls)
