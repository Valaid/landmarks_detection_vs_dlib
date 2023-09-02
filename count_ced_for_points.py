import argparse
import os
import os.path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def read_points(dir_path, max_points, mode='gt'):
    print('Reading directory {}'.format(dir_path))
    points = {}
    i = 0
    for idx, fname in enumerate(os.listdir(dir_path)):
        if max_points is not None and idx > max_points:
            break
                   

        cur_path = os.path.join(dir_path, fname)
        # TODO: add ability to exclude path
        #if os.path.isdir(cur_path):
        #  points.update(read_points(cur_path, max_points))

        if cur_path.endswith('.pts') or cur_path.endswith('.pts1'):
            if idx % 100 == 0:
                print(idx)
                
            if mode=='gt':            
                with open(cur_path) as cur_file:
                    lines = cur_file.readlines()
                    if lines[0].startswith('version'): # to support different formats
                        lines = lines[3:-1]
                    mat = np.fromstring(''.join(lines), sep=' ')
                    points[fname] = (mat[0::2], mat[1::2])
            else:
                with open(cur_path,'r') as cur_file:
                    lines = cur_file.readlines()
                    mat = np.fromstring(lines[0], sep=' ')
                    points[fname] = (mat[0::2], mat[1::2])
            

    return points


def count_ced(predicted_points, gt_points, path_crops):
    ceds = defaultdict(list)

    for method_name in predicted_points.keys():
        print('Counting ces. Method name {}'.format(method_name))
        for img_name in predicted_points[method_name].keys():
            if img_name in gt_points:
                #print('Processing key {}'.format(img_name))
                x_pred, y_pred = predicted_points[method_name][img_name]
                x_gt, y_gt = gt_points[img_name]
                n_points = x_pred.shape[0]
                assert n_points == x_gt.shape[0], '{} != {}'.format(n_points, x_gt.shape[0])

                w,h = cv2.imread('{}/{}'.format(path_crops,img_name.replace('.pts','.jpg'))).shape[:2]

                if 'HRnet' in method_name:
                    w,h = 1.25*w,1.25*h
                elif method_name=='onet':
                    w,h = 1.5*w,1.5*h
                normalization_factor = np.sqrt(h * w)

                diff_x = [x_gt[i] - x_pred[i] for i in range(n_points)]
                diff_y = [y_gt[i] - y_pred[i] for i in range(n_points)]
                dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
                avg_norm_dist = np.sum(dist) / (n_points * normalization_factor)
                ceds[method_name].append(avg_norm_dist)
                #print('Average distance for method {} = {}'.format(method_name, avg_norm_dist))
            else:
                print('Skipping key {}, because its not in the gt points'.format(img_name))
        ceds[method_name] = np.sort(ceds[method_name])

    return ceds


def count_ced_auc(errors):
    if not isinstance(errors, list):
        errors = [errors]

    aucs = []
    for error in errors:
        auc = 0
        proportions = np.arange(error.shape[0], dtype=np.float32) / error.shape[0]
        assert (len(proportions) > 0)

        step = 0.01
        for thr in np.arange(0.0, 1.0, step):
            gt_indexes = [idx for idx, e in enumerate(error) if e >= thr]
            if len(gt_indexes) > 0:
                first_gt_idx = gt_indexes[0]
            else:
                first_gt_idx = len(error) - 1
            auc += proportions[first_gt_idx] * step
        aucs.append(auc)
    return aucs


def main():
    parser = argparse.ArgumentParser(description='CED computation script',
                                     add_help=True)
    parser.add_argument('--gt_path', action='store', type=str, help='')
    parser.add_argument('--predictions_path', action='append', type=str, help='')
    parser.add_argument('--output_path', action='store', type=str, help='')
    parser.add_argument('--max_points_to_read', action='store', type=int, help='',
                        default=None)
    parser.add_argument('--dataset', action='store', type=str, help='',
                        default='300W')
    
    parser.add_argument('--error_thr', action='store', type=float, help='',
                        default=0.08)
    args = parser.parse_args()
    
    print('args.error_thr = {}'.format(args.error_thr))

    predicted_points = {}
    for pred_path in args.predictions_path:
        predicted_points['_'.join(pred_path.split('/')[-1].split('_')[1:])] = read_points(f'{pred_path}/{args.dataset}/test', args.max_points_to_read)
    gt_points = read_points(f'{args.gt_path}/{args.dataset}/test', args.max_points_to_read)
    #print(predicted_points.keys())
    #print(gt_points)

    ceds = count_ced(predicted_points, gt_points, f"{args.gt_path.replace('landmarks_task','landmarks_task_rects')}/{args.dataset}/test")

    # saving figure
    line_styles = [':', '-.', '--', '-']
    plt.figure(figsize=(30,20), dpi=100)
    for method_idx, method_name in enumerate(ceds.keys()):
        print('Plotting graph for the method {}'.format(method_name))
        err = ceds[method_name]
        proportion = np.arange(err.shape[0], dtype=np.float32) / err.shape[0]
        under_thr = err > args.error_thr
        last_idx = len(err)
        if len(np.flatnonzero(under_thr)) > 0:
            last_idx = np.flatnonzero(under_thr)[0]
        under_thr_range = range(last_idx)
        cur_auc = count_ced_auc(err)[0]

        plt.plot(err[under_thr_range], proportion[under_thr_range], label=method_name + ', auc={:1.3f}'.format(cur_auc),
             linestyle=line_styles[method_idx % len(line_styles)], linewidth=2.0)
    plt.title(f'{args.dataset}_test',fontsize=40)
    plt.legend(loc='right', prop={'size': 24})
    plt.savefig(args.output_path)


if __name__ == '__main__':
    main()
