"draw roc: fpr-tpr"

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    data = pd.read_csv(path,encoding='latin-1')
    print(data.head(2))

    tpr = data['TPR']
    fpr = data['FPR']

    return tpr,fpr


def roc(data):
    "data = [fpr_cnn,tpr_cnn,fpr_lstm,tpr_lstm]"
    marker_size = 18
    fontsize = 18

    plt.figure(num=1,figsize=(8, 6), dpi=150)
    if len(data) < 3:
        plt.plot(data[0],data[1],color='green', linestyle='--',linewidth=2, markeredgewidth=2, fillstyle='none',
                 markersize=marker_size,label='Non-Defended data')
    elif len(data) < 5:
        plt.plot(data[0],data[1],color='red', linestyle='-',linewidth=2, markeredgewidth=2, fillstyle='none',
                 markersize=marker_size,label='Non-Defended data')
        plt.plot(data[2], data[3], color='blue',  linestyle='--', linewidth=2, markeredgewidth=2,
                 fillstyle='none',
                 markersize=marker_size, label='Defended data')
    else:

        plt.plot(data[0],data[1],color='red', marker='*',linestyle='--',linewidth=2, markeredgewidth=2, fillstyle='none',
                 markersize=marker_size,label='WF-CNN')
        plt.plot(data[2],data[3],color='blue', marker='^',linestyle='--',linewidth=2, markeredgewidth=2, fillstyle='none',
                 markersize=marker_size,label='WF-LSTM')
        plt.plot(data[4],data[5],color='green', marker='.',linestyle='--',linewidth=2, markeredgewidth=2, fillstyle='none',
                 markersize=marker_size,label='DF-CNN')
    plt.xlabel('False Positive Rate',{'size':fontsize})
    plt.ylabel('True Positive Rate',{'size':fontsize})
    # plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.ylim(0,1)
    plt.tick_params(labelsize=fontsize)
    plt.legend(loc='lower right',fontsize=fontsize)

    plt.savefig('../fig/roc_E8_new.eps')
    plt.show()






def roc_main(flag=False):

    if flag:
        print('way 1 to draw roc (previous way)')
        "read csv"
        # classifiers = ['cnn','lstm','DF-CNN']  #[cnn,lstm]
        classifiers = ['DF-CNN']  #[DF-CNN]

        data = []
        for classifier in classifiers:
            path = '../result/roc/experiment8/ow_results_' + classifier + '.csv'
            # path = '../result/roc/experiment9/ow_results_' + classifier + '_gan.csv'
            tpr,fpr = load_data(path)
            data += [fpr,tpr]

        "draw roc"
        roc(data)

    else:
        print('way 2 to draw roc for DF-CNN (2021/09/21)')
        path1 = '../result/roc/experiment8/ow_results_DF-CNN.csv'
        path2 = '../result/roc/experiment9/ow_results_DF-CNN_gan.csv'
        data = []
        for p in [path1,path2]:
            tpr, fpr = load_data(p)
            data += [fpr, tpr]
        roc(data)





if __name__ == '__main__':
    roc_main()