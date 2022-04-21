"""
sampling certain samples of each class and write to csv
walkie talkie data
"""

from train import utils_wf

# path = '../data/wtf_pad/ow/train_WTFPAD.csv'
# outpath = '../data/wtf_pad/ow/adv_train_WTFPAD.csv'

# path = '../data/wtf_pad/ow/test_Mon_WTFPAD.csv'
# outpath = '../data/wtf_pad/ow/adv_test_Mon_WTFPAD.csv'

# path = '../data/wtf_pad/ow/test_Unmon_WTFPAD.csv'
# outpath = '../data/wtf_pad/ow/adv_test_UnMon_WTFPAD.csv'

path = '../data/wf/DF-CNN/DF-CNN_source_data/adv_train_WTFPAD.csv'
outpath = '../data/wf/DF-CNN/adv_train_WTFPAD.csv'


# path = '../data/wf/DF-CNN/DF-CNN_source_data/adv_test_WTFPAD.csv'
# outpath = '../data/wf/DF-CNN/adv_test_WTFPAD.csv'

# path = '../data/NoDef/ow/train_NoDef.csv'
# outpath = '../data/NoDef/ow/train_NoDef_sampled.csv'


X,Y = utils_wf.load_csv_data(path)
X_new, Y_new = [],[]

Y_set = set(Y)

if path == '../data/wtf_pad/ow/train_WTFPAD.csv' \
        or path == '../data/wf/DF-CNN/DF-CNN_source_data/adv_train_WTFPAD.csv' \
        or path == '../data/NoDef/ow/train_NoDef.csv':
    for y_set_i in Y_set:
        if y_set_i < 95:
            num = 331
        elif y_set_i == 95:
            num = 18000
        else:
            num = 0
            continue
        temp = y_set_i
        count = 0
        for i,y in enumerate(Y):
            if y == temp and count < num:
                X_new.append(X.iloc[i,:])
                Y_new.append(y)
                count += 1
                if count > num:
                    break


if path == '../data/wtf_pad/ow/test_Mon_WTFPAD.csv' or path == '../data/wf/DF-CNN/DF-CNN_source_data/adv_test_WTFPAD.csv':
    for y_set_i in Y_set:
        if y_set_i < 95:
            num = 92
        else:
            continue
        temp = y_set_i
        count = 0
        for i,y in enumerate(Y):
            if y == temp and count < num:
                X_new.append(X.iloc[i,:])
                Y_new.append(y)
                count += 1
                if count > num:
                    break

if path == '../data/wtf_pad/ow/test_Unmon_WTFPAD.csv':
    for y_set_i in Y_set:

        num = 6296
        temp = y_set_i
        count = 0
        for i,y in enumerate(Y):
            if y == temp and count < num:
                X_new.append(X.iloc[i,:])
                Y_new.append(y)
                count += 1
                if count > num:
                    break

    for i,y in enumerate(Y_new):
        if y == 100:
            Y_new[i] = 95

data = utils_wf.convert2dataframe(X_new,Y_new,mode='NoPadding')
utils_wf.write2csv(data,outpath)