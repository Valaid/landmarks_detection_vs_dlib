import os, subprocess
import pandas as pd
import glob
import numpy as np
import argparse
import torch
from tqdm import tqdm

def create_yaml():
    ''' create yaml file for HRnet'''

    with open('HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml','r') as f:
        config = f.readlines()
    
    config[8] = config[8].replace('face_landmarks_300w_valid.csv','face_landmarks_300w_test.csv')
    
    with open('HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml','w') as f:
        f.writelines(config)

def make_all_dirs(res_dir):
    datasets = ['300W','Menpo']
    modes = ['train','test']
    for dataset in datasets:
        for mode in modes:
            if not os.path.exists(f'data/{res_dir}/{dataset}/{mode}'):
                os.makedirs(f'data/{res_dir}/{dataset}/{mode}')

def prepare_csv(name_dataset):
    paths = glob.glob(f'data/landmarks_task_rects/{name_dataset}/test/*.jpg')
    paths.sort()
    points = []
    for path in tqdm(paths):
        with open(path.replace('.jpg','_rect_box.txt'),'r') as f:
            txt = f.read()
        box = np.fromstring(''.join(txt), sep=' ')
        x1,y1,x2,y2 = box
        w, h = x2 - x1, y2 - y1
        scale = max(w, h) / 200
        center_w = (x1 + x2) / 2
        center_h = (y1 + y2) / 2
        with open(path.replace('landmarks_task_rects','landmarks_task').replace('jpg','pts')) as cur_file:
            lines = cur_file.readlines()
            if lines[0].startswith('version'): # to support different formats
                lines = lines[3:-1]
            mat = np.fromstring(''.join(lines), sep=' ')
        points.append(np.hstack([np.array([path.replace('landmarks_task_rects','landmarks_task'),scale,center_w,center_h]),mat]))
        df_test = pd.DataFrame(points)
        df_test.to_csv('HRNet-Facial-Landmark-Detection/data/300w/face_landmarks_300w_test.csv',index=False)

def save_predicitons(predictions_name,res_dir):
    res = torch.load(f'HRNet-Facial-Landmark-Detection/output/300W/face_alignment_300w_hrnet_w18/predictions_{predictions_name}.pth')
    hrnet_df_test = pd.read_csv('HRNet-Facial-Landmark-Detection/data/300w/face_landmarks_300w_test.csv')
    for i in range(len(hrnet_df_test)):
        with open(hrnet_df_test.iloc[i][0].replace('.jpg','.pts').replace('landmarks_task',res_dir),'w') as f:
            f.write(' '.join(list(map(str,res[i].numpy().reshape(-1)))))

def main():
    parser = argparse.ArgumentParser(description='HRnet inference script',
                                     add_help=True)
    parser.add_argument('--dataset', action='store', type=str, help='', default='300W')
    parser.add_argument('--path_model_weights', action='store', type=str, 
                        help='Укажите путь из директории HRNet-Facial-Landmark-Detection', 
                        default='output/300W/face_alignment_300w_hrnet_w18')
    parser.add_argument('--prediction_name', help='model parameters', type=str,default='0')
    parser.add_argument('--result_dir', help='model parameters', type=str, default='result_HRnet')
    args = parser.parse_args()
    
    prepare_csv(args.dataset)
    create_yaml()
    if '.pth' in args.path_model_weights:
        model_weight_path = args.path_model_weights
    else:
        model_weight_path = f'{args.path_model_weights}/model_best_weights.pth'
    s = f'cd HRNet-Facial-Landmark-Detection && python tools/test.py \
        --cfg experiments/300w/face_alignment_300w_hrnet_w18.yaml \
            --model-file {model_weight_path} \
                --prediction_name {args.prediction_name}'
    print(s)
    p = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        #print('.', end='')
        # print(line)
        pass
    _ = p.wait()
    
    make_all_dirs(args.result_dir)
    save_predicitons(args.prediction_name,args.result_dir)


if __name__ == '__main__':
    main()