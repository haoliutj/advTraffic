import os
os.sys.path.append('..')

import torch
import torch.nn.functional as F
from train import models
from train import utils_wf,utils_shs
import os,sys
import time



class train_lstm:
    def __init__(self,opts):

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode + '/lstm'
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABEL: ', torch.cuda.is_available())

        if self.mode == 'wf or wf_ow':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')
        elif self.mode == 'detect':
            print('detecting adversary...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def train_model(self):

        if self.mode in ['wf','wf_ow','detect','wf_kf']:
            "load data"
            train_data = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'],shuffle=True)
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'],shuffle=True)

            "load target model structure"
            if self.mode == 'wf_ow':
                params = utils_wf.params_lstm_ow(self.opts['num_class'],self.opts['input_size'],self.opts['batch_size'])
            elif self.mode == 'wf_kf':
                params = utils_wf.params_lstm(self.opts['num_class'], self.opts['input_size'],self.opts['batch_size'])
            else:
                params = utils_wf.params_lstm(self.opts['num_class'],self.opts['input_size'],self.opts['batch_size'])
            target_model = models.lstm(params).to(self.device)
            target_model.train()

        elif self.mode == 'shs':
            "load data"
            train_data = utils_shs.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params_lstm(self.opts['num_class'],self.opts['input_size'],self.opts['batch_size'])
            target_model = models.lstm(params).to(self.device)
            target_model.train()

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        print(f"training data: {self.opts['train_data_path']}")
        print(f"testing data: {self.opts['test_data_path']}")

        "train process"
        optimizer = torch.optim.Adam(target_model.parameters(),lr=self.opts['lr'])

        for epoch in range(self.opts['epochs']):
            loss_epoch = 0
            for i, data in enumerate(train_data, 0):
                train_x, train_y = data
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                "batch_first = False"
                if not self.opts['batch_size']:
                    train_x = train_x.transpose(0,1)

                optimizer.zero_grad()
                logits_model = target_model(train_x)
                loss_model = F.cross_entropy(logits_model, train_y)
                loss_epoch += loss_model

                loss_model.backward(retain_graph=True)
                optimizer.step()

                if i % 100 == 0:
                    _, predicted = torch.max(logits_model, 1)
                    correct = int(sum(predicted == train_y))
                    accuracy = correct / len(train_y)
                    msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
                    print(msg.format(epoch, i, loss_model, accuracy))


            "save model every 10 epochs"
            if self.mode == 'wf_kf':
                model_name = '/target_model_%d.pth' % self.opts['id']
            else:
                model_name = '/target_model.pth'

            if epoch != 0 and epoch % 10 == 0:
                targeted_model_path = self.model_path + model_name
                torch.save(target_model.state_dict(), targeted_model_path)


        "test target model"
        target_model.eval()

        num_correct = 0
        total_instances = 0
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred_lab = torch.argmax(target_model(test_x), 1)
            num_correct += torch.sum(pred_lab == test_y, 0)
            total_instances += len(test_y)

        print('accuracy of target model against test dataset: %f\n' % (num_correct.item() / total_instances))



def main(opts):
    start_time = time.time()
    trainTargetModel = train_lstm(opts)
    trainTargetModel.train_model()
    end_time = time.time()
    running_time = end_time - start_time
    print(f'running time is {running_time:.3f} seconds')

def get_opts_wf(mode):
    return{
        'mode':mode,
        'num_class': 95,
        'input_size': 512,
        'batch_size': 64,
        'epochs':50,
        'lr': 0.006,
        'train_data_path': '../data/wf/train_NoDef_burst.csv',
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
    }


def get_opts_wf_kFold(mode,id):
    "website fingerprinting with 5-fold cross validation"
    return{
        'id': id,
        'mode':mode,
        'num_class': 95,
        'input_size': 512,
        'batch_size': 64,
        'epochs':50,
        'lr': 0.006,
        'train_data_path': '../data/wf/cross_val/traffic_train_%d.csv' % id,
        'test_data_path': '../data/wf/cross_val/traffic_test_%d.csv' % id,
    }


def get_opts_wf_ow(mode):
    return{
        'mode':mode,
        'num_class': 96,
        'input_size': 512,
        'batch_size': 64,
        'epochs':50,
        'lr': 0.004,
        'train_data_path': '../data/wf_ow/train_NoDef_mix.csv',
        'test_data_path': '../data/wf_ow/test_NoDef_Mon.csv',
    }

def get_opts_shs(mode):
    return{
        'mode':mode,
        'num_class': 101,
        'input_size': 256,
        'batch_size': 64,
        'epochs':50,
        'lr':0.001,
        'train_data_path': '../data/shs/traffic_train.csv',
        'test_data_path': '../data/shs/traffic_test.csv',
    }



def get_opts_detect(mode):
    "detect adversary"
    return {
        'mode': mode,
        'num_class': 4,
        'input_size': 512,
        'batch_size': 64,
        'epochs': 50,
        'lr': 0.006,
        'train_data_path': '../data/wf/cnn/adv_train_all.csv',
        'test_data_path': '../data/wf/cnn/adv_test_all.csv',
    }


if __name__ == '__main__':

    "wf_ow: website fingerprinting open-world setting"
    mode = 'wf'  # ['shs',wf','wf_ow','detect','wf-kf']
    print('mode: ', mode)
    if mode == 'wf':
        opts = get_opts_wf(mode)
    elif mode == 'wf_ow':
        opts = get_opts_wf_ow(mode)
    elif mode == 'shs':
        opts = get_opts_shs(mode)
    elif mode == 'detect':
        opts = get_opts_detect(mode)
    elif mode == 'wf_kf':
        k = 5  # num of K-Fold
        for id in range(k):
            print(f'{id}-Fold validation...')
            opts = get_opts_wf_kFold(mode, id)
            main(opts)
        print('K-Fold function completed, system is out')
        sys.exit()

    main(opts)