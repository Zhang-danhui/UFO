import copy
import hashlib
import importlib
import os
import pickle

import dgl
import numpy as np
import torch

from Backbones.gnns import GCN
from Backbones.utils import NodeLevelDataset, evaluate_batch
from dataset.utils import semi_task_manager
from training.utils import mkdir_if_missing


def build_noise_tag(args):
    if getattr(args, "label_noise", 0.0) <= 0:
        return ""
    p_str = f"{float(args.label_noise):.6g}".replace(".", "p")
    noise_type = getattr(args, "label_noise_type", "symmetric")
    type_tag = "" if noise_type == "symmetric" else f"_nt{noise_type}"
    return f"_nl{p_str}_s{int(getattr(args, 'seed', 1))}{type_tag}"


def apply_in_task_label_noise(labels, train_ids, ids_per_cls=None, noise_rate=0.0, generator=None, noise_type="symmetric"):
    p = float(noise_rate)
    if p <= 0 or train_ids is None:
        return labels, ids_per_cls
    p = max(0.0, min(1.0, p))

    device = labels.device
    train_ids_t = train_ids if torch.is_tensor(train_ids) else torch.as_tensor(train_ids, device=device, dtype=torch.long)
    n = int(train_ids_t.numel())
    if n == 0:
        return labels, ids_per_cls

    allowed = torch.unique(labels[train_ids_t]).tolist()
    if len(allowed) < 2:
        return labels, ids_per_cls

    n_flip = int(n * p)
    if n_flip <= 0:
        return labels, ids_per_cls

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)

    perm = torch.randperm(n, generator=generator, device=device)[:n_flip]
    flip_nodes = train_ids_t[perm]

    labels_noisy = labels.clone()
    L = torch.as_tensor(allowed, device=device, dtype=labels_noisy.dtype)
    old = labels_noisy[flip_nodes]

    if noise_type == "symmetric":
        idx_old = (old[:, None] == L[None, :]).float().argmax(dim=1)
        r = torch.randint(0, L.numel() - 1, (n_flip,), generator=generator, device=device)
        new_idx = r + (r >= idx_old).long()
        new = L[new_idx]
    elif noise_type == "pair_cycle":
        idx_old = (old[:, None] == L[None, :]).float().argmax(dim=1)
        new_idx = (idx_old + 1) % L.numel()
        new = L[new_idx]
    else:
        raise ValueError(f"Unsupported label noise type: {noise_type}")
    labels_noisy[flip_nodes] = new

    if ids_per_cls is not None:
        label2list = {}
        for i, ids in enumerate(ids_per_cls):
            if len(ids) == 0:
                continue
            yv = int(labels[ids[0]].item())
            label2list[yv] = i

        if len(label2list) >= 2:
            sets = [set(ids) for ids in ids_per_cls]
            flip_cpu = flip_nodes.detach().cpu().tolist()
            old_cpu = old.detach().cpu().tolist()
            new_cpu = new.detach().cpu().tolist()
            for nid, o, nn in zip(flip_cpu, old_cpu, new_cpu):
                oi = label2list.get(int(o), None)
                ni = label2list.get(int(nn), None)
                if oi is None or ni is None:
                    continue
                sets[oi].discard(int(nid))
                sets[ni].add(int(nid))
            ids_per_cls = [list(s) for s in sets]

    return labels_noisy, ids_per_cls


def make_task_file_name(args, task_cls, noise_tag):
    cls_str = str(task_cls)
    if len(cls_str) > 50:
        hash_tag = hashlib.md5(cls_str.encode()).hexdigest()[:10]
        return f"{args.dataset}_hash_{hash_tag}{noise_tag}.pkl"
    return f"{args.dataset}_{cls_str}{noise_tag}.pkl"


def data_prepare(args):
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    args.task_seq = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls - 1, args.n_cls_per_task)]
    args.n_tasks = len(args.task_seq)

    str_int_tsk = "inter_tsk_edge" if args.inter_task_edges else "no_inter_tsk_edge"

    for task, task_cls in enumerate(args.task_seq):
        noise_tag = "" if args.inter_task_edges else build_noise_tag(args)
        file_name = make_task_file_name(args, task_cls, noise_tag)
        file_path = f"{args.data_path}/{str_int_tsk}/{file_name}"

        try:
            if args.load_check:
                pickle.load(open(file_path, "rb"))
            elif file_name not in os.listdir(f"{args.data_path}/{str_int_tsk}"):
                pickle.load(open(file_path, "rb"))
        except:
            print(f"preparing data for task {task}")
            mkdir_if_missing(f"{args.data_path}/inter_tsk_edge")
            mkdir_if_missing(f"{args.data_path}/no_inter_tsk_edge")

            if args.inter_task_edges:
                cls_retain = []
                for clss in args.task_seq[0 : task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
                save_path = f"{args.data_path}/inter_tsk_edge/{file_name}"
                with open(save_path, "wb") as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                if getattr(args, "label_noise", 0.0) > 0:
                    labels_local = subgraph.dstdata["label"].squeeze()
                    g_noise = torch.Generator(device=labels_local.device)
                    g_noise.manual_seed(int(getattr(args, "seed", 1)) + int(task) + 12345)
                    labels_local, ids_per_cls = apply_in_task_label_noise(
                        labels_local,
                        train_ids,
                        ids_per_cls=ids_per_cls,
                        noise_rate=args.label_noise,
                        generator=g_noise,
                        noise_type=getattr(args, "label_noise_type", "symmetric"),
                    )
                    subgraph.ndata["label"] = labels_local.view(-1, 1)

                save_path = f"{args.data_path}/no_inter_tsk_edge/{file_name}"
                with open(save_path, "wb") as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)


def pipeline_task_IL_no_inter_edge_minibatch(args, valid=False):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    args.task_seq = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls - 1, args.n_cls_per_task)]
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()
    model = GCN(args).cuda(args.gpu)
    life_model = importlib.import_module(f"Baselines.{args.method}")
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)

    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split("/")[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f"{config_name}_{ite}_{task_cls}"
        save_model_path = f"{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl"
        n_cls_so_far += len(task_cls)

        noise_tag = build_noise_tag(args)
        file_name = make_task_file_name(args, task_cls, noise_tag)
        file_path = f"{args.data_path}/no_inter_tsk_edge/{file_name}"
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(file_path, "rb"))

        features, labels = subgraph.srcdata["feat"], subgraph.dstdata["label"].squeeze()
        if getattr(args, "label_noise", 0.0) > 0:
            if hasattr(subgraph.ndata, "__getitem__") and "_ID" in subgraph.ndata:
                global_train_ids = subgraph.ndata["_ID"][train_ids].long().cpu()
                if hasattr(dataset, "graph") and "label" in dataset.graph.ndata:
                    dataset.graph.ndata["label"][global_train_ids] = labels[train_ids].view(-1, 1).cpu()

        task_manager.add_task(task, n_cls_so_far)

        dataloader = dgl.dataloading.DataLoader(
            subgraph,
            train_ids,
            args.nb_sampler,
            batch_size=args.batch_size,
            shuffle=args.batch_shuffle,
            drop_last=False,
        )

        if valid:
            life_model_ins.train_task_and_fisher_no_val(args, subgraph, train_ids, task, dataset, prev_model=prev_model)

        if not valid:
            try:
                model = pickle.load(open(save_model_path, "rb")).cuda(args.gpu)
            except:
                model.load_state_dict(torch.load(save_model_path.replace(".pkl", ".pt")))

        acc_mean = []
        for t in range(task + 1):
            noise_tag = build_noise_tag(args)
            file_name = make_task_file_name(args, args.task_seq[t], noise_tag)
            file_path = f"{args.data_path}/no_inter_tsk_edge/{file_name}"
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(file_path, "rb"))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata["feat"], subgraph.dstdata["label"].squeeze()

            if t == 0:
                label_offset1 = 0
            else:
                label_offset1 = task_manager.get_label_offset(t - 1)[1]
            label_offset2 = task_manager.get_label_offset(t)[1]

            acc = evaluate_batch(
                args,
                model,
                subgraph,
                features,
                labels,
                test_ids,
                label_offset1,
                label_offset2,
                cls_balance=args.cls_balance,
                ids_per_cls=ids_per_cls_test,
            )

            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[: task + 1]
        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

        if valid:
            mkdir_if_missing(f"{args.result_path}/{subfolder_c}/val_models")
            try:
                with open(save_model_path, "wb") as f:
                    pickle.dump(model, f)
            except:
                torch.save(model.state_dict(), save_model_path.replace(".pkl", ".pt"))
        prev_model = copy.deepcopy(model).cuda()

    print("AP: ", acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print("AF: ", mean_backward)

    print("\n")
    return acc_mean, mean_backward, acc_matrix
