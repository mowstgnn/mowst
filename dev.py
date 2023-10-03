from utils import *
import torch
from train_test_engine import *
import wandb
import os
import numpy as np
import json

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.wandb_save_path):
        os.makedirs(args.wandb_save_path)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    res_save_path = "log3/"
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)

    if args.no_preset_struc is False:
        args = get_preset_struc(args)

    if args.log:
        wandb.init(project='moe_new_imp', config=args, dir=args.wandb_save_path)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                                    and not args.cpu else "cpu")
    print('Using {}'.format(device))

    if args.dataset in ['snap', 'arxiv-year']:
        args.directed = True

    verify_args(args)
    print(args.dataset)

    if args.setting == 'exp' or args.setting == 'hetero_exp':
        print('Experiment')
        dataset, data, split_idx = load_data(args)
        engine = get_engine(args, device, data, dataset, split_idx)
        result = engine.run_model(args.seed)
        res = print_info(args)
        print(res)
    elif args.setting == 'ten':
        dataset, data, split_idx = load_data(args)
        print('Ten run, ', args.method)
        if args.method == 'SimpleGate' or args.method == 'SimpleGateInTurn':

            if args.grid_range:
                lrs_large, dps_large, lrs, dps = grid_combo(args.dataset)
            else:
                lrs_large = [0.01, 0.001]
                dps_large = [0.5, 0.3, 0.1]
                lrs = [0.01, 0.001]
                dps = [0.5, 0.3, 0.1]
            for lr_large in lrs_large:
                for dp_large in dps_large:
                    for lr_gate in lrs:
                        for dropout_gate in dps:
                            args.lr = lr_large
                            args.dropout = dp_large
                            args.lr_gate = lr_gate
                            args.dropout_gate = dropout_gate
                            print(args.dataset, args.setting, args.submethod, args.lr, args.dropout, args.lr_gate,
                                  args.dropout_gate)
                            engine = get_engine(args, device, data, dataset, split_idx)
                            train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list, val_loss_list
                            for run_idx in range(args.run):
                                res = engine.run_model(args.seed + run_idx)

                                append_results(RES, res)
                            result_dict = create_result_dict(RES, args)

                            with open(res_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')


        elif args.method == 'vanilla':
            if args.model2 == "GPR-GNN":
                lrs = [.01, .05, .002]
                gpr_alphas = [.1, .2, .5, .9]
                hidden_dims = [16, 32, 64, 128, 256]
                for lr in lrs:
                    for gpr_alpha in gpr_alphas:
                        for hid in hidden_dims:
                            args.model2_hidden_dim = hid
                            args.lr = lr
                            args.gpr_alpha = gpr_alpha

                            args.run = 10
                            train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list, val_loss_list
                            for run_idx in range(args.run):
                                args.split_run_idx = run_idx
                                dataset, data, split_idx = load_data(args)
                                engine = get_engine(args, device, data, dataset, split_idx)
                                res = engine.run_model(args.seed)

                                append_results(RES, res)

                            result_dict = create_result_dict(RES, args)

                            with open(res_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json",
                                      "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')



            else:
                if args.dataset not in ['cora', 'citeseer', 'pubmed']:
                    lrs = [0.001, 0.01, 0.1]
                    # lrs = [0.01]

                    if args.dataset == "arxiv" and args.model2 == "Sage":
                        model2_hidden_dims = [512]
                        args.model1_num_lays = 4
                        args.model2_num_layers = 4
                    elif args.dataset == "product" and args.model2 == "Sage":
                        model2_hidden_dims = [512]
                        args.model1_num_layers = 4
                        args.model2_num_layers = 4
                    elif args.dataset == "flickr":
                        model2_hidden_dims = [256]
                        args.model2_num_layers = 3
                    else:
                        model2_hidden_dims = [4, 8, 12, 32]


                    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
                    for lr in lrs:
                        for dp in dropouts:
                        # for m2hd in model2_hidden_dims:
                            if args.dataset == 'snap-patents' and m2hd == 64:
                                break
                            args.lr = lr
                            args.dropout = dp
                            # args.model2_hidden_dim = m2hd
                            # args.model2_hidden_dim = m2hd
                            # args.model1_hidden_dim = m2hd

                            args.run = 10
                            train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list, val_loss_list
                            for run_idx in range(args.run):
                                args.split_run_idx = run_idx
                                dataset, data, split_idx = load_data(args)
                                engine = get_engine(args, device, data, dataset, split_idx)
                                res = engine.run_model(args.seed)

                                append_results(RES, res)

                            result_dict = create_result_dict(RES, args)

                            with open(res_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')
                else:
                    lrs = [0.1, 0.01, 0.001]
                    dps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    for lr in lrs:
                        for dp in dps:
                            args.lr = lr
                            args.dropout = dp
                            engine = get_engine(args, device, data, dataset, split_idx)
                            args.run = 10
                            train_list, valid_list, test_list, weight_list = [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list
                            for run_idx in range(args.run):
                                res = engine.run_model(args.seed + run_idx)

                                append_results(RES, res)
                            result_dict = create_result_dict(RES, args)

                            with open(res_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')
    elif args.setting == 'hetero_ten':
        print('Hetero Ten Run, ', args.method)
        if args.method == 'SimpleGate' or args.method == 'SimpleGateInTurn':
            if args.grid_range:
                lrs_large, dps_large, lrs, dps = grid_combo(args.dataset)
            else:
                lrs_large = [0.01, 0.001]
                dps_large = [0.5,  0.1]
                lrs = [0.01, 0.001]
                dps = [0.5, 0.1]
            for lr_large in lrs_large:
                for dp_large in dps_large:
                    for lr_gate in lrs:
                        for dropout_gate in dps:
                            args.lr = lr_large
                            args.dropout = dp_large
                            args.lr_gate = lr_gate
                            args.dropout_gate = dropout_gate
                            args.run = 5
                            train_list, valid_list, test_list, weight_list = [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list
                            for run_idx in range(args.run):
                                args.split_run_idx = run_idx
                                dataset, data, split_idx = load_data(args)
                                engine = get_engine(args, device, data, dataset, split_idx)
                                res = engine.run_model(args.seed)

                                append_results(RES, res)
                            result_dict = create_result_dict(RES, args)

                            with open(res_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')


        elif args.method == 'vanilla':

            lrs = [0.001, 0.01, 0.1]
            if args.dataset == "pokec":
                model2_hidden_dims = [4, 8, 12]
            else:
                model2_hidden_dims = [4, 8, 12, 32]
                # model2_hidden_dims = [32] # for sage
            # model2_hidden_dims = [0.1, 0.2, 0.3, 0.4, 0.5]
            for lr in lrs:
                for m2hd in model2_hidden_dims:
                    if args.dataset == 'snap-patents' and m2hd == 64:
                        break
                    args.lr = lr
                    args.model2_hidden_dim = m2hd
                    # args.model2_hidden_dim = m2hd
                    # args.model1_hidden_dim = m2hd

                    args.run = 5
                    train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                    RES = train_list, valid_list, test_list, weight_list, val_loss_list
                    for run_idx in range(args.run):
                        args.split_run_idx = run_idx
                        dataset, data, split_idx = load_data(args)
                        engine = get_engine(args, device, data, dataset, split_idx)
                        res = engine.run_model(args.seed)

                        append_results(RES, res)

                    result_dict = create_result_dict(RES, args)

                    with open(res_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                        json.dump(result_dict, f)
                        f.write('\n')
