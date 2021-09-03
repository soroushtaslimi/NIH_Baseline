import os
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

from data.custom_image_folder import MyImageFolder
from algorithm.base_algorithm import Base_Model

torch.backends.cudnn.benchmark = True
print(torch.cuda.is_available(), torch.cuda.device_count())

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization')
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=12, help='how many subprocesses to use for data loading')
parser.add_argument('--batch_size', type=int, help='size of each batch', default=128)
parser.add_argument('--cont', type=str, help='continue from a checkpoint', default="False")
parser.add_argument('--nih_img_size', type=int, help='image resized size in nih_lung dataset', default=128)
parser.add_argument('--model_type', type=str, help='base model to use: Densnet121, Resnet34, multihead_densenet121', default="Densenet121")
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--adjust_lr', type=int, help='if 0 uses scheduler.', default=0)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--dataset', type=str, help='nih', default='nih')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--save_epoch', type=int, help='number of epochs between saving models', default=5)
parser.add_argument('--pretrained', type=str, help='want pretrained model? True/False', default='False')
parser.add_argument('--center_crop', type=str, help='input images with center_crop? True/False', default='True')
parser.add_argument('--head_elements', type=int, help='number of neurons in start of each head', default=28)
parser.add_argument('--kfold', type=str, help='Uses The kfold train/test set', default='False')
parser.add_argument('--fold_id', type=int, help='which fold to use when using kfold setting', default=0)


args = parser.parse_args()
print('weight_decay:', args.weight_decay)

CHECKPOINT_PATH = '../Checkpoints/'
if args.kfold=='True':
    CHECKPOINT_PATH += '4fold/'
LOAD_PATH = '../Checkpoints/Densenet121_img128_forget_rateNone_lr0.001_batch76_epoch70.pth'

CHECKPOINT_PATH += 'validation/'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

random_seed = 33  # random.randint(0,100)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


if args.dataset == 'nih':
    input_channel = 3 if args.pretrained == 'True' or args.pretrained == 'true' else 1
    num_classes = 14
    # init_epoch = 0
    # filter_outlier = True
    # args.epoch_decay_start = 80
    # args.model_type = "resnet"
    # args.center_size = int(1024 * 0.75)

    data_path = '/mnt/sda1/project/nih-preprocess/Dataset/'  # TODO isnert data path
    if args.nih_img_size == 128:
        if args.center_crop == 'True':
            train_data_path = data_path + 'train2_crop0.75_resized128/'
            test_data_path = data_path + 'test2_crop0.75_resized128/'
        else:
            train_data_path = data_path + 'train2_resized128/'
            test_data_path = data_path + 'test2_resized128/'
    elif args.nih_img_size == 256:
        if args.center_crop == 'True':
            train_data_path = data_path + 'train2_crop0.75_resized256/'
            test_data_path = data_path + 'test2_crop0.75_resized256/'
        else:
            train_data_path = data_path + 'all_resized256/'
            valid_data_path = data_path + 'all_resized256/'
            test_data_path = data_path + 'all_resized256/'
    

    if args.kfold=='True':
        train_csv_path = '/mnt/sda1/project/nih-preprocess/Dataset/kfold_csv/4fold/fold' + str(args.fold_id) + '_train.csv'
        test_csv_path = '/mnt/sda1/project/nih-preprocess/Dataset/kfold_csv/4fold/fold' + str(args.fold_id) + '_test.csv'
    else:
        train_csv_path = 'processed_data_csv/train.csv'
        valid_csv_path = 'processed_data_csv/validation.csv'
        test_csv_path = 'processed_data_csv/test_without_nofinding.csv'

    if args.pretrained == 'True' or args.pretrained == 'true':
        img_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        img_transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        img_transform = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    # transforms.CenterCrop(args.center_size),
                                    # transforms.Resize((args.nih_img_size, args.nih_img_size)),  
                                    # transforms.RandomAffine(degrees=20, translate = (0.05, 0.05)),
                                    transforms.ToTensor()])

        img_transform_test = transforms.Compose([
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])


    train_dataset = MyImageFolder(
        root=train_data_path,
        csv_path=train_csv_path,
        transform=img_transform
    )

    valid_dataset = MyImageFolder(
        root=valid_data_path,
        csv_path=valid_csv_path,
        transform=img_transform_test
    )

    test_dataset = MyImageFolder(
        root=test_data_path,
        csv_path=test_csv_path,
        transform=img_transform_test
    )


def save_models(base_model, epoch):
    checkpoint = {
        'epoch': epoch,
        'model': base_model.model.state_dict(),
        'optimizer': base_model.optimizer.state_dict(),
        'scheduler': base_model.scheduler
    }
    checkpoint_name = (CHECKPOINT_PATH + args.model_type + 
                                            ('_pretrained' if args.pretrained == 'True' else '') +
                                            ('_noCenterCrop' if args.center_crop == 'False' else '') +
                                            '_img' + str(args.nih_img_size) + 
                                            (('_forget_rate'+str(args.forget_rate)) if args.forget_rate is not None else '')+
                                            '_lr' + str(args.lr) +
                                            (('_weightDecay'+str(args.weight_decay)) if args.weight_decay != 0.0 else '') +
                                            '_batch' + str(args.batch_size) +
                                            '_epoch' + str(epoch) + '.pth')
    torch.save(checkpoint, checkpoint_name)


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True,
                                               pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False,
                                              pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False,
                                              pin_memory=True)
    
    print('len(train_loader.dataset):', len(train_loader.dataset))

    train_dataset_array = next(iter(train_loader))[0]
    print('train_dataset_array.shape:', train_dataset_array.shape)
    
    # Define models
    print('building model...')

    start_epoch = 0

    start_checkpoint = None
    if args.cont == 'True' or args.cont == 'true':
        start_checkpoint = torch.load(LOAD_PATH, map_location=device)
        start_epoch = start_checkpoint['epoch']

    # num test samples: len(test_loader) * args.batch_size
    model = Base_Model(args, train_dataset, device, input_channel, num_classes, start_checkpoint=start_checkpoint)

    del start_checkpoint
    
    # evaluate models with random weights
    test_auc = model.evaluate(test_loader)

    print(
        'Epoch [%d/%d] Test Accuracy on the %s test images:' % (
            start_epoch + 1, args.n_epoch, len(test_dataset)))
    print('Model1 AUCs:', test_auc)
    print('Model1 mean AUC:', sum(test_auc)/len(test_auc))
    

    # training
    for epoch in range(start_epoch + 1, args.n_epoch):
        # train models
        model.train(train_loader, epoch)

        # evaluate models
        valid_auc = model.evaluate(valid_loader)

        print(
            'Epoch [%d/%d] Validation Accuracy on the %s test images:' % (
                epoch + 1, args.n_epoch, len(valid_dataset)))
        print('Model1 AUCs:', valid_auc)
        print('Model1 mean AUC:', sum(valid_auc)/len(valid_auc))


        test_auc = model.evaluate(test_loader)

        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images:' % (
                epoch + 1, args.n_epoch, len(test_dataset)))
        print('Model1 AUCs:', test_auc)
        print('Model1 mean AUC:', sum(test_auc)/len(test_auc))

        if epoch % args.save_epoch == 0:
            save_models(model, epoch)


if __name__ == '__main__':
    main()
