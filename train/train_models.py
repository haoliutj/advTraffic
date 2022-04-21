"""
train models: DF_CNN and Var-CNN (not include) based on website fingerprinting

wf: website fingerprinting
wf_ow: website fingerprinting open world
wf-kf: website fingerprinting k-fold validation
"""

import os
os.sys.path.append('..')

import torch
import torch.nn.functional as F
from train import models
from train import utils_wf
import time
from sklearn.metrics import accuracy_score




def evaluate_step(model,data_iterator,criterion,device):
    """
    validation step
    :param model:
    :param data_iterator:
    :param criterion:
    :param device:
    :return: avg_loss and avg_acc
    """
    epoch_loss = 0
    epoch_acc = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in data_iterator:
            X,y = batch
            X,y = X.to(device), y.to(device)

            preds = model(X)
            loss = criterion(preds,y)

            preds_labels = torch.argmax(preds,dim=1)
            acc = accuracy_score(preds_labels.cpu().numpy(),y.cpu().numpy())

            epoch_loss += loss.item()
            epoch_acc += acc
    avg_loss = epoch_loss/len(data_iterator)
    avg_acc = epoch_acc/len(data_iterator)

    return avg_loss,avg_acc


def train_step(model,data_iterator,optimizer,criterion,device):
    """
    training step by backward process
    :param model:
    :param data_iterator:
    :param optimizer:
    :param criterion:
    :param device:
    :return:
    """
    epoch_loss, epoch_acc = 0,0

    model.to(device)
    model.train()

    for batch in data_iterator:
        X,y = batch
        X,y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds,y)

        preds_labels = torch.argmax(preds,dim=1)
        acc = accuracy_score(preds_labels.cpu().numpy(),y.cpu().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss += loss
        epoch_acc += acc

    avg_loss = epoch_loss / len(data_iterator)
    avg_acc = epoch_acc / len(data_iterator)

    return avg_loss,avg_acc


def epoch_time(start,end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs


def train_loop(epochs,model,train_data,val_data,lr,model_out_path,device,test=False):
    """

    :param epochs:
    :param model:
    :param train_data:
    :param val_data:
    :param lr:
    :param model_out_path: path for saving trained model
    :param device:
    :param test: True --> test trained model on test data, otherwise not
    :return:
    """

    best_val_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1,epochs+1):

        start_time = time.time()

        train_loss,train_acc = train_step(model,train_data,optimizer,criterion,device)
        val_loss,val_acc= evaluate_step(model,val_data,criterion,device)

        end_time = time.time()

        epoch_min,epoch_sec = epoch_time(start_time,end_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),model_out_path)
            print(f"chechpoint saved at {model_out_path}")

        print(f"Epoch: {epoch:02}, Epoch Time {epoch_min}m{epoch_sec}s, Train Loss {train_loss:.3f} Train Acc {train_acc:.3f}, Val Loss {val_loss:.3f} Val Acc {val_acc:.3f}")

    "test step is needed"
    if test:
        tes_data_monitored = utils_wf.load_data_main(opts['test_data_path'],64,shuffle=True)
        if opts['mode']:
            test_data_unmonitored = utils_wf.load_data_main(opts['test_unmon_data_path'],64,shuffle=True)
            test_loss_unmon, test_acc_unmon = evaluate_step(model, test_data_unmonitored, criterion, device)

        test_loss, test_acc = evaluate_step(model,tes_data_monitored,criterion,device)
        print(f"-------------- acc on test data --------------")
        print(f"Monitored Data Test Loss {test_loss:.3f} Monitored Data Test Acc {test_acc:.3f}")
        if opts['mode']:
            print(f"Unmonitored Data Test Loss {test_loss_unmon:.3f} unmonitored Data Test Acc {test_acc_unmon:.3f}")


def main_train(opts):

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {device}')

    # load data
    train_data = utils_wf.load_data_main(opts['train_data_path'], opts['batch_size'], shuffle=True)
    val_data = utils_wf.load_data_main(opts['val_data_path'], opts['batch_size'], shuffle=False)
    test_data = utils_wf.load_data_main(opts['test_data_path'], opts['batch_size'], shuffle=False)

    # load model
    model = models.DF_CNN(opts['num_class'])

    if not os.path.exists(opts['checkpoint']):
        os.mkdir(opts['checkpoint'])
    model_save_path = '%s%s_model.pth' % (opts['checkpoint'],opts['target_model'])

    train_loop(opts['epochs'],model,train_data,val_data,opts['lr'],model_save_path,device,test=True)


def get_opts(mode=None):
    "parameters for data after self processed"
    return {
        # 'model':mode,
        'target_model': 'DF_CNN',
        'checkpoint': '../model/wf/DF-CNN/',
        'batch_size': 512,
        'epochs': 200,
        'lr': 0.0001,
        'num_class': 95,
        'train_data_path': '../data/wf/with_validation/train_NoDef_burst.csv',
        'val_data_path': '../data/wf/with_validation/val_NoDef_burst.csv',
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
    }


def get_opts_wf_ow(mode='ow'):
    "params in the open-world setting"
    return{
        'mode':mode,
        'target_model': 'DF_CNN',
        'checkpoint': '../model/wf_ow/DF-CNN/',
        'batch_size': 512,
        'epochs': 200,
        'lr': 0.0001,
        'num_class':96,
        'train_data_path': '../data/wf_ow/with_validation/train_NoDef_burst.csv',
        'val_data_path': '../data/wf_ow/with_validation/val_NoDef_burst.csv',
        'test_data_path': '../data/wf_ow/test_NoDef_Mon.csv',
        'test_unmon_data_path': '../data/wf_ow/test_NoDef_UnMon.csv',
    }


def get_opts_source(mode=None):
    "parameters for source data from DeepFingerprinting paper without processing"
    return {
        # 'mode':mode,
        'target_model': 'DF_CNN',
        'checkpoint': '../model/wf/DF-CNN_source_data/',
        'batch_size': 128,
        'epochs': 200,
        'lr': 0.0001,
        'num_class': 95,
        'train_data_path': '../data/NoDef/csv_source_data/train_NoDef.csv',
        'val_data_path': '../data/NoDef/csv_source_data/valid_NoDef.csv',
        'test_data_path': '../data/NoDef/csv_source_data/test_NoDef.csv',
    }



if __name__ == '__main__':

    opts = get_opts()
    # opts = get_opts_wf_ow()

    main_train(opts)



