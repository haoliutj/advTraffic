"""
sampling certain samples of each class and write to csv
walkie talkie data
"""

from train import utils_wf

# path = '../data/WalkieTalkie/ow/train_WalkieTalkie.csv'
# outpath = '../data/WalkieTalkie/ow/adv_train_WalkieTalkie.csv'

# path = '../data/WalkieTalkie/ow/test_Mon_WalkieTalkie.csv'
# outpath = '../data/WalkieTalkie/ow/adv_test_Mon_WalkieTalkie.csv'

# path = '../data/WalkieTalkie/ow/test_Unmon_WalkieTalkie.csv'
# outpath = '../data/WalkieTalkie/ow/adv_test_UnMon_WalkieTalkie.csv'

# path = '../data/WalkieTalkie/cw/from_DF_paper/test_WalkieTalkie.csv'
# outpath = '../data/WalkieTalkie/cw/from_DF_paper/adv_test_WalkieTalkie.csv'

path = '../data/WalkieTalkie/cw/from_DF_paper/train_WalkieTalkie.csv'
outpath = '../data/WalkieTalkie/cw/from_DF_paper/adv_train_WalkieTalkie.csv'


X,Y = utils_wf.load_csv_data(path)
X_new, Y_new = [],[]

Y_set = set(Y)

if path == '../data/WalkieTalkie/ow/train_WalkieTalkie.csv' \
       or path == '../data/WalkieTalkie/cw/from_DF_paper/train_WalkieTalkie.csv':
    for y_set_i in Y_set:
        if y_set_i < 95:
            num = 331
        elif y_set_i == 100:
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

    for i,y in enumerate(Y_new):
        if y == 100:
            Y_new[i] = 95

if path == '../data/WalkieTalkie/ow/test_Mon_WalkieTalkie.csv' \
        or path == '../data/WalkieTalkie/cw/from_DF_paper/test_WalkieTalkie.csv':
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

if path == '../data/WalkieTalkie/ow/test_Unmon_WalkieTalkie.csv':
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