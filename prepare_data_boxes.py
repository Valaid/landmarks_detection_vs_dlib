import argparse
import dlib
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

def iou(box1, box2, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    inter = np.clip(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0, np.inf) * \
            np.clip(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0, np.inf)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou

def convert_and_trim_bb(image, rect):
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, rect.left())
	startY = max(0, rect.top())
	endX = min(rect.right(), image.shape[1])
	endY = min(rect.bottom(), image.shape[0])
	return (startX, startY, endX, endY)

def make_all_dirs(res_dir):
    datasets = ['300W','Menpo']
    modes = ['train','test']
    for dataset in datasets:
        for mode in modes:
            print(f'data/{res_dir}/{dataset}/{mode}')
            if not os.path.exists(f'data/{res_dir}/{dataset}/{mode}'):
                os.makedirs(f'data/{res_dir}/{dataset}/{mode}')

def delete_little_points_imgs(paths):
    for path in paths:
        with open(path.replace('jpg','pts'),'r') as f:
            text = f.read()
        # print(path)
        if int(text.split('\n')[1].split(' ')[-1])!=68:
            os.remove(path)
            os.remove(path.replace('.pts','.jpg'))

def max_box_fr_points(path):
    with open(path,'r') as f:
        text = f.read()
    lst_points = []
    for xy in text.split('\n')[3:-2]:
        lst_points.append(list(map(float,xy.split(' '))))
    points_arr = np.array(lst_points)
    x1,y1 = np.min(points_arr,axis=0)
    x2,y2 = np.max(points_arr,axis=0)
    return [x1,y1,x2,y2]

def prepare_dlib(data_dir,iou_threshold):
    make_all_dirs(f'{data_dir.split("/")[-1]}_rects')
    paths = glob.glob(f'{data_dir}/*/*/*.jpg')
    delete_little_points_imgs(paths)
    detector = dlib.get_frontal_face_detector()
    for path in tqdm(paths):
        lst_box=[]
        img = dlib.load_rgb_image(path)
        dets = detector(img, 1)
        if len(dets):
            if len(dets)>1:
                point_box = max_box_fr_points(path.replace('jpg','pts'))
                for det in dets:
                    lst_box.append(convert_and_trim_bb(img,det))
                box_arr = np.array(lst_box)
                ious = iou(point_box,box_arr)
                if max(ious)>iou_threshold:
                    main_box = lst_box[np.argmax(ious)]
                else:
                    continue
            else:
                main_box = convert_and_trim_bb(img,dets[0])
            new_path = path.replace(f'{data_dir.split("/")[-1]}',f'{data_dir.split("/")[-1]}_rects')
            print('!!!!!!!!!!'+new_path)
            img = cv2.imread(path)
            cv2.imwrite(new_path, img[main_box[1]:main_box[3],main_box[0]:main_box[2]])
            with open(new_path.replace('.jpg','_rect_box.txt'),'w') as f:
                f.write(' '.join(list(map(str,main_box))))
def main():
    parser = argparse.ArgumentParser(description='HRnet inference script',
                                     add_help=True)
    parser.add_argument('--data_dir', action='store', type=str, help='', default='data/lanmarks_task')
    parser.add_argument('--iou_threshold', action='store', type=float, help='', default=0.3)
    args = parser.parse_args()

    prepare_dlib(args.data_dir,args.iou_threshold)

if __name__ == '__main__':
    main()