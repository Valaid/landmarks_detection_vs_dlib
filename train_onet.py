import argparse
import datetime
import os
from model import ONet
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from clearml import Task, Logger
import numpy as np
import glob
import pandas as pd
import cv2
from dataset import ImageDataset
from sklearn.model_selection import train_test_split

torch.manual_seed(2023)

def make_all_dirs(res_dir):
    datasets = ['300W','Menpo']
    modes = ['train','test']
    for dataset in datasets:
        for mode in modes:
            if not os.path.exists(f'data/{res_dir}/{dataset}/{mode}'):
                os.makedirs(f'data/{res_dir}/{dataset}/{mode}')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def test(model_path, lst_paths, test_dataset, res_dir, batch_size=64, num_workers=8, use_cuda=True):
    
    make_all_dirs(res_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    net = ONet(is_train=True)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    if use_cuda:
        net.cuda()
    with torch.no_grad():
        for batch_idx,(image,gt_landmark,meta) in enumerate(tqdm(test_loader)):

            if use_cuda:
                image = image.cuda()
                # gt_landmark = gt_landmark.cuda()

            output = net(image)

            score_map = output.data.cpu()
            
            for n in range(score_map.size(0)):
                real_xy = score_map[n, :].numpy() * meta['new_size'][n].numpy() + np.tile([meta['left_corner_xy'][0][n],meta['left_corner_xy'][1][n]],68)
                with open(lst_paths[meta['index'][n].item()].replace('.jpg','.pts').replace('landmarks_task',res_dir),'w') as f:
                    f.write(' '.join(list(map(str,real_xy.tolist()))))
                
def train_onet(train_dataset, val_dataset, end_epoch,
               batch_size, base_lr=0.01, step_sceduler=25,
               num_workers=4, use_cuda=True, model_store_path='Onet_train'):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)
        
    task = Task.init(
        project_name='vlab', 
        task_name='landmarks', 
        tags=['Onet'])
    
    log = Logger.current_logger()
    
    loss_train_history = []
    loss_val_history = []

    losses_train = AverageMeter()
    losses_val = AverageMeter()
    
    net = ONet(is_train=True)
    net.train()
    if use_cuda:
        net.cuda()
    
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_sceduler)
    best_loss = 1000000000

    for cur_epoch in range(end_epoch):
        landmark_loss_list=[]
        net.train()
        for batch_idx,(image,gt_landmark, _) in enumerate(tqdm(train_loader)):

            if use_cuda:
                image = image.cuda()
                gt_landmark = gt_landmark.cuda()

            landmark_offset_pred = net(image)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)
            loss = criterion(gt_landmark,landmark_offset_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses_train.update(loss.item(), image.size(0))

        net.eval()
        with torch.no_grad():
            for batch_idx,(image,gt_landmark,_) in enumerate(val_loader):
                
                if use_cuda:
                    image = image.cuda()
                    gt_landmark = gt_landmark.cuda()
                    
                output = net(image)

                # score_map = output.data.cpu()
                # loss
                loss = criterion(output, gt_landmark)

                losses_val.update(loss.item(), image.size(0))
                
        loss_train_history.append(losses_train.avg)
        loss_val_history.append(losses_val.avg)
        
        log.report_scalar("Landmark loss", "Val", iteration=batch_idx, value=losses_val.avg)
        log.report_scalar("Landmark loss", "Train", iteration=batch_idx, value=losses_train.avg)
        lr_scheduler.step()

        print("Epoch: %d, landmark loss train: %s, landmark loss val: %s " % (cur_epoch, loss_train_history[-1],loss_val_history[-1]))
        if loss_val_history[-1]<best_loss:
            best_loss = loss_val_history[-1]
            torch.save(net.state_dict(), f"{model_store_path}/best_epoch.pt")
        # torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))
    return loss_train_history, loss_val_history

def prepare_data(paths, scale=0.5, max_outbox_points=1):
    points = {}
    for path in paths:
        with open(path.replace('landmarks_task_rects','landmarks_task').replace('jpg','pts')) as cur_file:
            lines = cur_file.readlines()
            if lines[0].startswith('version'): # to support different formats
                lines = lines[3:-1]
            mat = np.fromstring(''.join(lines), sep=' ')
            points[path.replace('landmarks_task_rects','landmarks_task')] = mat
    df = pd.DataFrame.from_dict(points, orient='index')
    array_boxes = []
    for path in df.index:
        with open(path.replace('landmarks_task','landmarks_task_rects').replace('.jpg','_rect_box.txt'),'r') as f:
            txt = f.read()
        array_boxes.append(np.fromstring(''.join(txt), sep=' '))
    df[['x1','y1','x2','y2']] = np.array(array_boxes)
    lst_lenghts_outpoints = []
    for i in tqdm(range(len(df))):
        key_points = df.iloc[i].values[:-4].reshape(68,2)
        x1,y1,x2,y2 = df.iloc[i].values[-4:]
        img_size = cv2.imread(df.iloc[i].name).shape
        size = max(y2-y1,x2-x1)
        added_frame = size*scale/2
        lst_lenghts_outpoints.append(len(set(np.hstack((np.argwhere(key_points[:,0]<max(0,x1-added_frame)).squeeze(),np.argwhere(key_points[:,0]>max(img_size[1],x2+added_frame)).squeeze(),
                                                                    np.argwhere(key_points[:,1]<min(0,y1-added_frame)).squeeze(),np.argwhere(key_points[:,1]>max(img_size[0],y2+added_frame)).squeeze())))))
    df['points_out'] = lst_lenghts_outpoints
    res_df = df[df['points_out']<=max_outbox_points]
    return res_df
    
    
def prepare_data_train(scale=0.5, max_outbox_points=1):
    paths = glob.glob('data/landmarks_task_rects/*/train/*.jpg')
    df_prepared = prepare_data(paths,scale,max_outbox_points)
    df_train, df_test = train_test_split(df_prepared,test_size=0.2,random_state=20)
    train_dataset = ImageDataset(df_train,(48,48))
    val_dataset = ImageDataset(df_test,(48,48))
    return train_dataset,val_dataset


def prepare_data_test(dataset, scale, max_outbox_points):
    paths = glob.glob(f'data/landmarks_task_rects/{dataset}/test/*.jpg')
    df_test = prepare_data(paths,scale,max_outbox_points)
    test_dataset = ImageDataset(df_test,(48,48))
    return test_dataset, df_test.index

def main():
    parser = argparse.ArgumentParser(description='Onet train and inference script',
                                     add_help=True)
    parser.add_argument('--mode', action='store', type=str, help='', default='train')
    parser.add_argument('--scale', action='store', type=float, help='',default=0)
    parser.add_argument('--output_path', action='store', type=str, help='', default='Onet_train')
    parser.add_argument('--cuda', action='store_true',help='')
    parser.add_argument('--max_outbox_points', action='store', type=int, help='',
                        default=1)
    parser.add_argument('--batch_size', action='store', type=int, help='',
                        default=16)
    parser.add_argument('--num_workers', action='store', type=int, help='',
                        default=4)
    parser.add_argument('--max_epoch', action='store', type=int, help='',
                        default=50)
    parser.add_argument('--result_dir', action='store', type=str, help='',
                        default='result_onet')
    parser.add_argument('--base_lr', action='store', type=str, help='',
                        default=0.001)
    parser.add_argument('--step_scheduler', action='store', type=int, help='',
                        default=25)
    parser.add_argument('--dataset', action='store', type=str, help='',
                        default='300W')
    args = parser.parse_args()
    
    if args.mode=='train':
        train_dataset,val_dataset = prepare_data_train(args.scale,args.max_outbox_points)
        _, _ = train_onet(train_dataset, val_dataset, end_epoch=args.max_epoch,
                          batch_size=args.batch_size,base_lr=args.base_lr,step_sceduler=args.step_scheduler,
                          num_workers=args.num_workers,use_cuda=args.cuda, model_store_path=args.output_path)
        
    if args.mode=='inference':
        test_dataset,paths  = prepare_data_test(args.dataset,args.scale,args.max_outbox_points)
        test(f'{args.output_path}/best_epoch.pt', paths, test_dataset, args.result_dir, batch_size=args.batch_size, num_workers=args.num_workers,use_cuda=args.cuda)
        

if __name__ == '__main__':
    main()