# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pathlib
import random
import time

import pandas as pd
import torch
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import *

from models.random_model import RandomModel
from run_custom import run_xLSTM_prediction, run_xLSTM_reconstruct, run_xLSTM_reconstruct_soft_dtw, \
    run_xLSTM_prediction_soft_dtw

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-M/')
    parser.add_argument('--file_lsit', type=str, default='../Datasets/File_List/TSB-AD-M-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/multi/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/multi/')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--AD_Name', type=str, default='RandomModel')
    args = parser.parse_args()

    models_dir = pathlib.Path('saved_models') / args.AD_Name
    models_dir.mkdir(parents=True, exist_ok=True)
    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok=True)
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_lsit)['file_name'].values
    if args.AD_Name in Optimal_Multi_algo_HP_dict:
        Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
    else:
        Optimal_Det_HP = {}

    save_path = pathlib.Path(f'{args.save_dir}/{args.AD_Name}.csv')
    if save_path.exists():
        already_existing = pd.read_csv(save_path)
    else:
        already_existing = None

    write_csv = []
    for filename in file_list:
        if os.path.exists(target_dir + '/' + filename.split('.')[0] + '.npy'): continue
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        # print('data: ', data.shape)
        # print('label: ', label.shape)

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]
        model = None

        start_time = time.time()
        if args.AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
        elif args.AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
        elif args.AD_Name == 'xLSTMAD-F':
            output, model = run_xLSTM_prediction(data_train, data)
        elif args.AD_Name == 'xLSTMAD-F-DTW':
            output, model = run_xLSTM_prediction_soft_dtw(data_train, data)
        elif args.AD_Name == 'xLSTMAD-R':
            output, model = run_xLSTM_reconstruct(data_train, data)
        elif args.AD_Name == 'xLSTMAD-R-DTW':
            output, model = run_xLSTM_reconstruct_soft_dtw(data_train, data)
        elif args.AD_Name == 'RandomModel':
            output = RandomModel().decision_function(data)
        else:
            raise Exception(f"{args.AD_Name} is not defined")

        end_time = time.time()
        run_time = end_time - start_time

        if isinstance(output, np.ndarray):
            logging.info(
                f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
            np.save(target_dir + '/' + filename.split('.')[0] + '.npy', output)
        else:
            logging.error(f'At {filename}: ' + output)

        if model is not None:
            torch.save(model.state_dict(), models_dir / f'{filename.split(".")[0]}_{args.AD_Name}.pt')

        ### whether to save the evaluation result
        if args.save:
            pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except Exception as e:
                print('Error while processing results: ', e)
                list_w = [0] * 9
            list_w.insert(0, run_time)
            list_w.insert(0, filename)
            write_csv.append(list_w)

            ## Temp Save
            col_w = list(evaluation_result.keys())
            col_w.insert(0, 'Time')
            col_w.insert(0, 'file')
            w_csv = pd.DataFrame(write_csv, columns=col_w)
            if already_existing is not None:
                w_csv = pd.concat([already_existing, w_csv], ignore_index=True)
            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)
