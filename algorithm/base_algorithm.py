# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.nn import DataParallel
from sklearn.metrics import roc_auc_score

import numpy as np
from model.Densenet import densenet121
from model.resnet import resnet34
from .base_loss import loss_base
from model.multihead import Multihead

from torchsummary import summary

class Base_Model:
    def __init__(self, args, train_dataset, device, input_channel, num_classes, start_checkpoint=None):
        self.num_classes = num_classes
        # self.num_test_samples = num_test_samples
        self.num_data_loader_samples = {}

        self.class_weights = None
        
        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr

        # define drop rate schedule
        if args.forget_rate is None:
            self.rate_schedule = [None] * args.n_epoch
        else:
            self.rate_schedule = np.ones(args.n_epoch) * args.forget_rate
            self.rate_schedule[:args.num_gradual] = np.linspace(0, args.forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.print_freq = args.print_freq
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        pretrained = args.pretrained == 'True' or args.pretrained == 'true'

        if args.model_type == "Densenet121":
            if pretrained:
                self.model = torchvision.models.densenet121(pretrained=True)
                num_ftrs = self.model.classifier.in_features
                self.model.classifier = nn.Linear(num_ftrs, num_classes)
            else:
                self.model = densenet121(num_classes=num_classes)
        elif args.model_type == "Resnet34":
            if pretrained:
                raise NotImplementedError()
            else:
                self.model = resnet34(num_classes=num_classes)
        elif args.model_type == "multihead_densenet121":
            self.model = Multihead(num_classes, head_elements=args.head_elements, base_model='densenet121', pretrained=pretrained)

        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)


        self.model.to(device)
        # summary(self.model, (input_channel, args.nih_img_size, args.nih_img_size))
        print(self.model.parameters)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor= 0.5, patience=3, mode='min')
        # self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_weights)
        self.loss_fn = loss_base
        self.adjust_lr = args.adjust_lr

        if start_checkpoint is not None:
            self.model.load_state_dict(start_checkpoint['model'])
            self.optimizer.load_state_dict(start_checkpoint['optimizer'])
            self.scheduler = start_checkpoint['scheduler']


    def num_data_samples(self, data_loader):
        if data_loader in self.num_data_loader_samples:
            return self.num_data_loader_samples[data_loader]
        res = 0
        for images, labels in data_loader:
            res += len(labels)
        self.num_data_loader_samples[data_loader] = res
        return res


    def evaluate_model(self, data_loader, model):
        model.eval()  # Change model to 'eval' mode.

        num_samples = self.num_data_samples(data_loader)

        all_outputs = np.empty((num_samples, self.num_classes), float)
        all_labels = np.empty((num_samples, self.num_classes), np.int8)

        cur_ind = 0

        for images, labels in data_loader:
            images = images.to(self.device)
            logits = model(images)
            # outputs = F.softmax(logits, dim=1)  # for crossentropy loss
            outputs = torch.sigmoid(logits)  # for BCEloss

            all_outputs[cur_ind:cur_ind+len(labels), :] = outputs.cpu().detach().numpy()
            all_labels[cur_ind:cur_ind+len(labels), :] = labels.cpu().detach().numpy()

            cur_ind += len(labels)
        

        assert cur_ind == num_samples
        auc = roc_auc_score(all_labels, all_outputs, average=None)  # alloutputs for multiLabel classification, alloutputs[:, 1] for binary
        return auc


    # Evaluate the Model
    def evaluate(self, data_loader):
        print('Evaluating ...')
        auc1 = self.evaluate_model(data_loader, self.model)
        if self.adjust_lr == 0:
            self.scheduler.step(-sum(auc1)/len(auc1))
        return auc1


    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model.train()  # Change model to 'train' mode

        # mean_loss = 0.0
        # cnt_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward + Backward + Optimize
            logits = self.model(images)

            loss_1 = self.loss_fn(logits, labels, forget_rate=self.rate_schedule[epoch], class_weights=self.class_weights)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d], Loss: %.6f, learning_rate: %.6f'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size,
                       loss_1.data.item(), self.optimizer.param_groups[0]['lr'])
                )
                # mean_loss += loss_1.data.item()
                # cnt_loss += 1
        
        # mean_loss /= cnt_loss
        # if self.adjust_lr == 0:
        #     self.scheduler.step(mean_loss)
                

