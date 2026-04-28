import copy
import os

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn.nets.myresnet import ResidualNet
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import RandomPermutation, ReversePermutation

class NET(torch.nn.Module):
    def __init__(self, model, task_manager, args):
        super(NET, self).__init__()
        self.reg = args.UFO_args['memory_strength']
        self.task_manager = task_manager
        self.current_task = -1
        self.prev_net = None
        self.prev_flow = None
        self.beta_kd = args.beta_kd
        self.beta_logit_kd = args.beta_logit_kd
        self.T_kd = args.T_kd
        self.label_noise = args.label_noise
        self.net = model

        self.lamb = args.lamb
        self.ce = torch.nn.functional.cross_entropy
        self.global_step = 0
        self.fisher = {}
        self.optpar = {}
        self.epochs = 0
        self.classify_criterion_noreduce = torch.nn.NLLLoss(reduction='none')

        self.num_classes = args.n_cls
        self.xa_dim = args.d_data

        self.flow = self.get_1d_nflow_model(
            feature_dim=self.xa_dim,
            hidden_feature=512,
            context_feature=self.num_classes,
            num_layers=4,
        ).to(device=f'cuda:{args.gpu}')

        self.flow_optimizer = torch.optim.Adam(
            self.flow.parameters(),
            lr=1e-4,
            weight_decay=args.weight_decay
        )
        self.opt = torch.optim.Adam(
            (p for p in self.net.parameters() if p.requires_grad),
            lr=args.lr * 0.1,
            weight_decay=args.weight_decay
        )

        self.k_loss_flow = 0.1
        self.k_flow_lastflow = 0.2
        self.flow_explore_theta = 0.2
        self.flow_weight_alpha = 0.2
        self.flow_weight_min = 0.5
        self.flow_weight_max = 1.5
        self.flow_infonce_weight = 0.05
        self.flow_infonce_temp = 0.1
        self.flow_n_negatives = 8

    def forward_to_preW_batch(self, blocks, input_features):
        b0 = blocks[0].local_var()
        x = input_features * b0.srcdata['out_norm']
        b0.srcdata['x'] = x
        b0.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'x_agg'))
        m_preW = b0.dstdata['x_agg'] * b0.dstdata['in_norm']
        return m_preW

    def get_1d_nflow_model(self, feature_dim, hidden_feature, context_feature, num_layers):
        transforms = []
        for l in range(num_layers):
            if l < 2:  
                transforms.append(ReversePermutation(features=feature_dim))
            else:    
                transforms.append(RandomPermutation(features=feature_dim))


            mask = (torch.arange(0, feature_dim) >= (feature_dim // 2)).float()
            net_func = lambda in_d, out_d: ResidualNet(
                in_features=in_d,
                out_features=out_d,
                hidden_features=hidden_feature,
                context_features=context_feature,
                num_blocks=2,
                activation=F.leaky_relu,
                dropout_probability=0
            )
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=net_func))

        transform = CompositeTransform(transforms)
        base_dist = StandardNormal(shape=[feature_dim])
        return Flow(transform, base_dist)

    def sample_from_flow(self, flow, labels, batch_size, device):
        label = np.random.choice(labels, batch_size)
        onehot = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        onehot[np.arange(batch_size), label] = 1.0
        onehot = torch.tensor(onehot, device=device)
        flow_xa = flow.sample(num_samples=1, context=onehot).squeeze(1)
        return flow_xa.detach(), torch.tensor(label, device=device).long(), onehot.detach()

    def flow_absolute_weight(self, xa, labels):
        with torch.no_grad():
            onehot = F.one_hot(labels, num_classes=self.num_classes).float()
            logp = self.flow.log_prob(inputs=xa, context=onehot)
            w_raw = torch.exp(logp - logp.max())
            weight = w_raw / w_raw.mean()
            weight = 1.0 + self.flow_weight_alpha * (weight - 1.0)
            weight = torch.clamp(weight, min=self.flow_weight_min, max=self.flow_weight_max)
        return weight.detach()

    def train_a_batch_flow(self, xa, labels, last_flow=None, seen_max_class=None):
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        loss_data = -self.flow.log_prob(inputs=xa, context=y_onehot).mean()

        loss_last_flow = 0.0
        if last_flow is not None:
            assert seen_max_class is not None and seen_max_class > 0
            with torch.no_grad():
                flow_xa, _, flow_onehot = self.sample_from_flow(
                    last_flow,
                    labels=list(range(seen_max_class)),
                    batch_size=xa.size(0),
                    device=xa.device
                )
            loss_last_flow = -self.flow.log_prob(inputs=flow_xa, context=flow_onehot).mean()
            loss_last_flow = self.k_flow_lastflow * loss_last_flow

        loss = loss_data + loss_last_flow
        self.flow_optimizer.zero_grad()
        loss.backward()
        self.flow_optimizer.step()
        return loss_data.detach()
        
    def observe_task_IL_batch(self, args, g, dataloader, features, labels,
                              t, train_ids, ids_per_cls, dataset):

        self.epochs += 1
        self.net.train()

        if t == 0:
            offset1, offset2 = 0, self.task_manager.get_label_offset(t)[1]
        else:
            offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]

        k_cur = offset2 - offset1

        device = f'cuda:{args.gpu}'
        if t != self.current_task:
            if t > 0:
                self.prev_net = copy.deepcopy(self.net).to(device)
                self.prev_net.eval()
                for p in self.prev_net.parameters():
                    p.requires_grad_(False)

                self.prev_flow = copy.deepcopy(self.flow).to(device)
                self.prev_flow.eval()
            self.current_task = t

        last_flow = self.prev_flow if (t > 0 and self.prev_flow is not None) else None
        last_total_loss = torch.tensor(0.0, device=device)

        for _, _, blocks in dataloader:
            blocks = [b.to(device=device) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            self.flow.train()
            with torch.no_grad():
                xa_new = self.forward_to_preW_batch(blocks, input_features)
            xa_new = xa_new.detach()

            if t > 0 and offset1 > 0 and last_flow is not None:
                old_labels = list(range(0, offset1))
                xa_old2, y_old_global2, _ = self.sample_from_flow(
                    last_flow,
                    labels=old_labels,
                    batch_size=xa_new.size(0),
                    device=xa_new.device
                )
                xa_mix = torch.cat([xa_new, xa_old2.detach()], dim=0)
                y_mix = torch.cat([output_labels, y_old_global2.detach()], dim=0)
            else:
                xa_mix = xa_new
                y_mix = output_labels

            self.train_a_batch_flow(
                xa_mix, y_mix,
                last_flow=last_flow,
                seen_max_class=offset1
            )

            self.net.zero_grad()
            self.opt.zero_grad()

            logits_new = self.net.forward_batch(blocks, input_features, task=t)
            y_new_local = output_labels - offset1
            probs = F.softmax(logits_new, dim=1)
            loss_new_per = F.nll_loss(torch.log(probs), y_new_local, reduction='none')

            w_new = self.flow_absolute_weight(xa_new, output_labels)
            loss_new = (loss_new_per * w_new).mean()

            loss_replay = torch.tensor(0.0, device=logits_new.device)
            if t > 0 and offset1 > 0 and last_flow is not None:
                old_labels = list(range(0, offset1))
                with torch.no_grad():
                    xa_old, y_old_global, old_onehot = self.sample_from_flow(
                        last_flow,
                        labels=old_labels,
                        batch_size=logits_new.size(0),
                        device=logits_new.device
                    )

                    log_probs = last_flow.log_prob(inputs=xa_old, context=old_onehot)
                    w_raw = torch.exp(log_probs - log_probs.max())
                    weights = w_raw / w_raw.mean()

                xa_old = xa_old.detach()
                y_old_global = y_old_global.detach()
                weights = weights.detach()

                k = args.n_cls_per_task
                old_task = (y_old_global // k).long()
                old_y_local = (y_old_global % k).long()

                for tt in torch.unique(old_task).tolist():
                    tt = int(tt)
                    idx = (old_task == tt)
                    if idx.sum() == 0:
                        continue
                    xa_old_h = F.relu(self.net.gcn_layers[0].linear(xa_old[idx]))
                    logits_tt = self.net.forward_from_xa(xa_old_h, task=tt)
                    loss_per_sample = F.cross_entropy(logits_tt, old_y_local[idx], reduction='none')
                    weighted_loss = loss_per_sample * weights[idx]
                    loss_replay = loss_replay + weighted_loss.mean()

                num_old_tasks = max(1, len(torch.unique(old_task)))
                loss_replay = loss_replay / num_old_tasks

            loss_kd_feat = torch.tensor(0.0, device=logits_new.device)
            loss_kd_logit = torch.tensor(0.0, device=logits_new.device)

            if t > 0 and self.prev_net is not None:
                was_training = self.net.training
                self.net.eval()
                xa_student = self.net.forward_to_xa_batch(blocks, input_features)
                with torch.no_grad():
                    xa_teacher = self.prev_net.forward_to_xa_batch(blocks, input_features)

                loss_kd_feat = F.mse_loss(xa_student, xa_teacher)

                kd_sum = 0.0
                kd_cnt = 0
                for tt in range(t):
                    logits_s_tt = self.net.forward_from_xa(xa_student, task=tt)
                    with torch.no_grad():
                        logits_t_tt = self.prev_net.forward_from_xa(xa_teacher, task=tt)
                    kd_sum = kd_sum + self.kd_logits(logits_s_tt, logits_t_tt, T=self.T_kd)
                    kd_cnt += 1
                if kd_cnt > 0:
                    loss_kd_logit = kd_sum / kd_cnt
                self.net.train(mode=was_training)

            loss_main = (
                loss_new
                + self.lamb * loss_replay
                + self.beta_kd * loss_kd_feat
                + self.beta_logit_kd * loss_kd_logit
            )

            loss_main.backward()
            self.opt.step()

            last_total_loss = loss_main.detach()

        return last_total_loss

    def kd_logits(self, logits_s, logits_t, T=2.0):
        p_t = F.softmax(logits_t / T, dim=1)
        log_p_s = F.log_softmax(logits_s / T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

    def train_task_and_fisher_no_val(self, args, subgraph, train_ids, task, dataset, prev_model=None):
        device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        self.net.to(device)
        self.flow.to(device)

        train_loader = dgl.dataloading.DataLoader(
            subgraph,
            train_ids,
            args.nb_sampler,
            batch_size=args.batch_size,
            shuffle=args.batch_shuffle,
            drop_last=False
        )

        for epoch in range(args.epochs):
            self.observe_task_IL_batch(
                args, subgraph, train_loader, None, None, task,
                train_ids, None, dataset
            )
            torch.cuda.empty_cache()
