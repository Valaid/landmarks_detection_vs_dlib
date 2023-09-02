import os, subprocess
import pandas as pd
import glob
from sklearn.model_selection import train_test_split


def create_yaml():
    ''' create yaml file for HRnet'''
    # path = os.path.abspath(os.getcwd()) + '/data'

    with open('HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml','r') as f:
        config = f.readlines()
    
    config[8] = config[8].replace('face_landmarks_300w_test.csv','face_landmarks_300w_valid.csv')
    
    
    text = START_YAML_TEXT%(path, train_path, val_path, nc, str(names))
    
    with open('HRNet-Facial-Landmark-Detection/experiments/300w/face_alignment_300w_hrnet_w18.yaml','w') as f:
        f.writelines(config)
        
def prepare_csv():
    paths = glob.glob('data/landmarks_task_rects/*/train/*.jpg')
    paths.sort()
    points = []
    for path in paths:
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
        df_trainval = pd.DataFrame(points)
        df_train, df_val = train_test_split(df_trainval,test_size=0.2,random_state=20)
        df_train.to_csv('HRNet-Facial-Landmark-Detection/data/300w/face_landmarks_300w_train.csv',index=False)
        df_val.to_csv('HRNet-Facial-Landmark-Detection/data/300w/face_landmarks_300w_valid.csv',index=False)
    
def main():
    
    prepare_csv()
    create_yaml()
    s = 'cd HRNet-Facial-Landmark-Detection && python tool/train.py --cfg experiments/300w/face_alignment_300w_hrnet_w18.yaml'
    p = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # заготовка для вывода отладочной информации при обучении
    for line in p.stdout.readlines():
        #print('.', end='')
        # print(line)
        pass
    _ = p.wait()
    

        

if __name__ == '__main__':
    main()