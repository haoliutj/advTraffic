"draw recall-precision"

import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    """
    load data
    """
    data = pd.read_csv(path)
    print(data.head(2))

    return data


def recall_precision(data,outpath,plot_all=False):
    """
    [Precision-fgsm,Recall-fgsm,Precision-deepfool,Recall-deepfool,
    Precision-pgd,Recall-pgd, Precision-GAN,Recall-GAN]
    """

    # header = ['P_fgsm','R_fgsm','P_deepfool','R_deepfool','P_pgd','R_pgd','P_GAN','R_GAN']
    header = ['P_fgsm','R_fgsm','P_deepfool','R_deepfool','P_pgd','R_pgd','P_GAN','R_GAN',
              'P_walkie_talkie','R_walkie_talkie','P_wtf-pad','R_wtf-pad','P_mockingbird','R_mockingbird']

    marker_size = 18
    fontsize = 18

    plt.figure(figsize=(8, 6), dpi=150)
    ## x axis: recall; y axis: precision
    if len(header) == 8 or plot_all==True:
        plt.plot(data[header[1]], data[header[0]], color='red', marker='+', linestyle='--', linewidth=2, markeredgewidth=2,
                 fillstyle='none', markersize=marker_size,label='FGSM-AdvTraffic')
        plt.plot(data[header[3]], data[header[2]], color='blue', marker='^', linestyle='--', linewidth=2, markeredgewidth=2,
                 fillstyle='none', markersize=marker_size,label='DeepFool-AdvTraffic')
        plt.plot(data[header[5]], data[header[4]], color='green', marker='o', linestyle='--', linewidth=2, markeredgewidth=2,
                 fillstyle='none', markersize=marker_size,label='PGD-AdvTraffic')
        plt.plot(data[header[7]], data[header[6]], color='black', marker='*', linestyle='--', linewidth=2, markeredgewidth=2,
                 fillstyle='none', markersize=marker_size,label='AdvGAN-AdvTraffic')
    if len(header) > 8:
        if not plot_all:
            plt.plot(data[header[7]], data[header[6]], color='black', marker='*', linestyle='--', linewidth=2,
                     markeredgewidth=2,
                     fillstyle='none', markersize=marker_size, label='AdvTraffic-AdvGAN')
        plt.plot(data[header[9]], data[header[8]], color='orange', marker='2', linestyle='--', linewidth=2,
                 markeredgewidth=2,fillstyle='none', markersize=marker_size, label='Walkie-Talkie')
        plt.plot(data[header[11]], data[header[10]], color='lime', marker='1', linestyle='--', linewidth=2,
                 markeredgewidth=2,fillstyle='none', markersize=marker_size, label='WTF-PAD')
        plt.plot(data[header[13]], data[header[12]], color='gray', marker='3', linestyle='--', linewidth=2,
                 markeredgewidth=2,fillstyle='none', markersize=marker_size, label='Mockingbird')

    plt.xlabel('Recall', {'size': fontsize})
    plt.ylabel('Precision', {'size': fontsize})
    plt.tick_params(labelsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)

    plt.savefig(outpath)
    plt.show()



def main(flag=False):

    if flag:
        print('ros in previous way')
        classifiers = ['cnn','lstm']  #[cnn,lstm]
        for classifier in classifiers:
            path = '../result/roc/experiment9/ow_results_' + classifier + '.csv'
            data = load_data(path)
            outpath = '../fig/rec_pre_%s.eps' % classifier
            recall_precision(data,outpath)
    else:
        print('ros for DF-CNN')
        classifier = 'DF-CNN'
        path = '../result/roc/experiment9/ow_results_' + classifier + '_new.csv'
        data = load_data(path)
        outpath = '../fig/rec_pre_%s_new.eps' % classifier
        recall_precision(data, outpath)




if __name__ == '__main__':
    main()