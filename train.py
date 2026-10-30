import argparse
import copy
import os
import pickle
import random
import shutil
import traceback
import warnings
from distutils.util import strtobool

from pipeline import *
from training.utils import assign_hyp_param, compose_hyper_params, mkdir_if_missing, remove_illegal_characters, set_seed, str2dict

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CoraFull-CL", choices=["CoraFull-CL", "Photo-CL", "CS-CL", "WikiCS-CL"], help="Supported datasets: CoraFull-CL, Photo-CL, CS-CL, WikiCS-CL")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--seed", type=int, default=1, help="seed for exp")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs, default = 200")
    parser.add_argument("--lamb", type=float, default=0.0410059679049203)
    parser.add_argument("--beta_kd", type=float, default=0.09027265899430625)
    parser.add_argument("--beta_logit_kd", type=float, default=0.0654227579168154)
    parser.add_argument("--T_kd", type=float, default=2.71897807211061)
    parser.add_argument("--cls_replay_ratio", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--backbone", type=str, default="GCN", choices=["GCN"], help="backbone GNN")
    parser.add_argument("--method", type=str, choices=["UFO"], default="UFO", help="baseline continual learning method")
    parser.add_argument("--share-labels", type=strtobool, default=False, help="task-IL specific, whether to share output label space for different tasks")
    parser.add_argument("--inter-task-edges", type=strtobool, default=False, help="whether to keep the edges connecting nodes from different tasks")
    parser.add_argument("--classifier-increase", type=strtobool, default=True, help="(deprecated) class-IL specific, whether to enlarge the label space with the coming of new classes, unrealistic to be set as False")
    parser.add_argument("--refresh_data", type=strtobool, default=False, help="whether to load existing splitting or regenerate")
    parser.add_argument("--d_dtat", default=None, help="will be assigned during running")
    parser.add_argument("--n_cls", default=None, help="will be assigned during running")
    parser.add_argument("--ratio_valid_test", nargs="+", default=[0.2, 0.2], help="ratio of nodes used for valid and test")
    parser.add_argument("--transductive", type=strtobool, default=True, help="using transductive or inductive")
    parser.add_argument("--default_split", type=strtobool, default=False, help="whether to  use the data split provided by the dataset")
    parser.add_argument("--task_seq", default=[])
    parser.add_argument("--n-task", type=int, default=0, help="will be assigned during running")
    parser.add_argument("--n_cls_per_task", type=int, default=7, help="how many classes does each task  contain")
    parser.add_argument("--GCN-args", default={"h_dims": [256], "dropout": 0.0, "batch_norm": False})
    parser.add_argument("--UFO_args", type=str2dict, default={"memory_strength": 10000.})
    parser.add_argument("--cls-balance", type=strtobool, default=True, help="whether to balance the cls when training and testing")
    parser.add_argument("--repeats", type=int, default=1, help="how many times to repeat the experiments for the mean and std")
    parser.add_argument("--ILmode", default="taskIL", choices=["taskIL"])
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--label-noise", type=float, default=0.0, help="within-task symmetric label noise rate p in [0,1]; for each task, relabel p fraction of TRAIN nodes uniformly to other labels within the same task")
    parser.add_argument("--label-noise-type", type=str, default="symmetric", choices=["symmetric", "pair_cycle"], help="label noise type applied on training nodes only")
    parser.add_argument("--minibatch", type=strtobool, default=True, help="whether to use the mini-batch training")
    parser.add_argument("--batch_shuffle", type=strtobool, default=True, help="whether to shuffle the data when constructing the dataloader")
    parser.add_argument("--n_nbs_sample", type=lambda x: [int(i) for i in x.replace(" ", "").split(",")], default=[10, 25], help="number of neighbors to sample per hop, use comma to separate the numbers when using the command line, e.g. 10,25 or 10, 25")
    parser.add_argument("--replace_illegal_char", type=strtobool, default=False)
    parser.add_argument("--ori_data_path", type=str, default="/store/data", help="the root path to raw data")
    parser.add_argument("--data_path", type=str, default="./data", help="the path for processed data")
    parser.add_argument("--result_path", type=str, default="./results", help="the path for saving results")
    parser.add_argument("--load_check", type=strtobool, default=False, help="whether to check the existence of processed data by loading")
    parser.add_argument("--perform_testing", type=strtobool, default=True, help="whether to perform final testing")
    return parser
def run_experiment(args):
    args.ratio_valid_test = [float(i) for i in args.ratio_valid_test]
    # torch.set_num_threads(1)
    set_seed(args)
    args._split_rng_consumed = False
    mkdir_if_missing(f"{args.data_path}")

    method_args = {"UFO": args.UFO_args}
    backbone_args = {"GCN": args.GCN_args}
    hyp_param_list = compose_hyper_params(method_args[args.method])

    AP_best, name_best = 0, None
    AP_dict = {str(hyp_params).replace("'", "").replace(" ", "").replace(",", "_").replace(":", "_"): [] for hyp_params in hyp_param_list}
    AF_dict = {str(hyp_params).replace("'", "").replace(" ", "").replace(",", "_").replace(":", "_"): [] for hyp_params in hyp_param_list}
    PM_dict = {str(hyp_params).replace("'", "").replace(" ", "").replace(",", "_").replace(":", "_"): [] for hyp_params in hyp_param_list}

    for hyp_params in hyp_param_list:
        hyp_params_str = str(hyp_params).replace("'", "").replace(" ", "").replace(",", "_").replace(":", "_")
        assign_hyp_param(args, hyp_params)

        args.n_nbs_sample = args.n_nbs_sample[:1]
        args.nb_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.n_nbs_sample)

        main = pipeline_task_IL_no_inter_edge_minibatch
        train_ratio = round(1 - args.ratio_valid_test[0] - args.ratio_valid_test[1], 2)
        subfolder = f"no_inter_task_edges/tsk_IL/train_ratio_{train_ratio}/"

        name = f"{subfolder}val_{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(hyp_params.values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{args.repeats}"
        if args.minibatch:
            name = name + f"_bs{args.batch_size}"
        mkdir_if_missing(f"{args.result_path}/" + subfolder)
        if args.replace_illegal_char:
            name = remove_illegal_characters(name)

        if os.path.isfile(f"{args.result_path}/{name}.pkl"):
            print("The results of the following configuration exists, will load this result for use \n", f"{args.result_path}/{name}.pkl")
            with open(f"{args.result_path}/{name}.pkl", "rb") as f:
                saved_acc_matrices = pickle.load(f)
            AP = float(np.mean([np.mean(matrix[-1]) for matrix in saved_acc_matrices]))
            if AP > AP_best:
                AP_best = AP
                hyp_best_str = hyp_params_str
                name_best = name
                print(f"best params is {hyp_best_str}, best AP is {AP_best}")
            continue

        acc_matrices = []
        for ite in range(args.repeats):
            args.current_model_save_path = [name, ite]
            try:
                AP, AF, acc_matrix = main(args, valid=True)
                AP_dict[hyp_params_str].append(AP)
                acc_matrices.append(acc_matrix)
                torch.cuda.empty_cache()
                if ite == 0:
                    with open(f"{args.result_path}/log.txt", "a") as f:
                        f.write(name)
                        log_info = (
                            f"\n[Params] lamb: {args.lamb}, "
                            f"beta_kd: {args.beta_kd}, "
                            f"beta_logit_kd: {args.beta_logit_kd}, "
                            f"T_kd: {args.T_kd} "
                            f"-> [Result] AP: {AP:.4f}, AF: {AF:.4f}\n"
                        )
                        f.write(log_info)
                        print(log_info)
            except Exception as e:
                mkdir_if_missing(f"{args.result_path}/errors/" + subfolder)
                if ite > 0:
                    name_ = f"{subfolder}val_{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(hyp_params.values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{ite}"
                    with open(f"{args.result_path}/{name_}.pkl", "wb") as f:
                        pickle.dump(acc_matrices, f)
                print("error", e)
                name = "errors/{}".format(name)
                acc_matrices = traceback.format_exc()
                print(acc_matrices)
                print("error happens on \n", name)
                torch.cuda.empty_cache()
                break

        if np.mean(AP_dict[hyp_params_str]) > AP_best:
            AP_best = np.mean(AP_dict[hyp_params_str])
            hyp_best_str = hyp_params_str
            name_best = name
        with open(f"{args.result_path}/{name}.pkl", "wb") as f:
            pickle.dump(acc_matrices, f)

    config_name = name_best.split("/")[-1]
    subfolder_c = name_best.split(config_name)[-2]
    if args.perform_testing:
        print("----------Now in testing--------")
        acc_matrices = []
        for ite in range(args.repeats):
            args.current_model_save_path = [name_best, ite]
            AP_test, AF_test, acc_matrix_test = main(args, valid=False)
            acc_matrices.append(acc_matrix_test)
        with open(f"{args.result_path}/{name_best}.pkl".replace("val", "te"), "wb") as f:
            pickle.dump(acc_matrices, f)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)
