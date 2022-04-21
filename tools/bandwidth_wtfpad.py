import os
os.sys.path.append('..')

from train import utils_wf
import numpy as np

path_1 = '../data/NoDef/ow/train_NoDef_sampled.csv'
# path_2 = '../data/wtf_pad/ow/adv_train_WTFPAD.csv'
path_2 = '../data/WalkieTalkie/ow/adv_train_WalkieTalkie.csv'

# path_1 = '../data/wf/with_validation/train_NoDef_burst.csv'
# path_2 = '../data/wf/DF-CNN/adv_train_WTFPAD.csv'

x1,y1 = utils_wf.load_csv_data(path_1)
x2,y2 = utils_wf.load_csv_data(path_2)

ori = np.sum(abs(np.array(x1)))
pertb = np.sum(abs(np.array(x2)))
noise = abs(pertb - ori)
bandwidth = noise / ori


print(f'WTF-PAD bandwidth overhead: {bandwidth}')