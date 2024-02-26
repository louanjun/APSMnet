import os
import scipy.io as sio

import sys
#sys.path.append('/home/zilong/SSTN')     # add the SSTN root path to environment path

import torch
import utils
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.utils.data import DataLoader

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from osgeo import gdal
import time
import collections
import logging
import argparse

import torch
from torch.utils import data
from lion_pytorch.lion_pytorch import Lion
from sklearn.decomposition import PCA
from sklearn import metrics, preprocessing
from utils import cal_results, predVisIN
import collections
import pandas as pd

#from NetworksBlocks import  SSNet_AEAE_PC, SSRN 
from coatnet2 import coatnet_6
from coatnet_aux import coatnet_7,coatnet_8
from DCSN import HyperCLR
from APSMnet import APSMnet1,APSMnet2
#from APSMnet_CA import APSMnet_CA1,APSMnet_CA2
from APSMnet_Drop import APSMnet_Drop1,APSMnet_Drop2
from ResNet import resnet18
from vit import vit

class HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples, labels):
        
        self.list_IDs = list_IDs
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        y = self.labels[ID]

        return X, y
def read_tiff(tiff_path):
    dataset = gdal.Open(tiff_path)
    height, width = dataset.RasterYSize, dataset.RasterXSize
    geotransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    image = dataset.ReadAsArray()
    #image = np.array(data,dtype='float')  # 转化为numpy数
    image[np.isnan(image)] = 0  # 替换nan为0

    del dataset
    return geotransform, proj, image, width, height    

def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:,range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, :, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix

def rSampling(groundTruth, sample_num):              #divide dataset into train and test datasets
    whole_loc = {}
    train = {}
    val = {}
    m = np.max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        whole_loc[i] = indices
        train[i] = indices[:sample_num[i]]
        val[i] =  indices[sample_num[i]:]
    whole_indices = []
    train_indices = []
    val_indices = []
    for i in range(m):
        whole_indices += whole_loc[i]
        train_indices += train[i]
        val_indices += val[i]
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)        
    return whole_indices, train_indices, val_indices
        

if __name__=="__main__": 

    parser = argparse.ArgumentParser("2022")
    year = 2000
    # parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    # parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='init learning rate')
    # parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    # parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    # parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    # parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    # parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    # parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    # parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    # parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--sample', type=int, default=4000, help='sample sizes for training')
    parser.add_argument('--model', type=str, default='APSMnet_Drop_2aux', help='select network to train')  
    #'Coatnet_aux'   'ResNet18'  'Coatnet'  'Vit' '3DCSN' 'APSMnet_Drop_2aux'  'APSMnet' 'APSMnet_2aux' 'APSMnet_Drop'
    
    #parser.add_argument('--phi', type=str, default='AEAE', help='sequential order of network')
    # parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    # parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    # parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    # parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
    # parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)

    np.random.seed(2)
    cudnn.benchmark = True
    torch.manual_seed(2)
    cudnn.enabled=True
    torch.cuda.manual_seed(2)

   # args.save = 'E:/lajcode/GEE_DL/train_model/2000-train-sample{}-model-{}-arch-{}-{}-lr{}'.format(args.sample ,args.model, args.phi, time.strftime("%Y%m%d-%H%M%S"), args.learning_rate)
    rote = 'D:/code/swin_transformer/'
    args.save = rote+'train_model/{}-train-sample{}-model-{}-{}-lr{}'.format(year, args.sample ,args.model,time.strftime("%Y%m%d-%H%M%S"), args.learning_rate)
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('train_2000.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('Start Time:')
    logging.info(time.strftime("%Y%m%d-%H%M%S"))


    IN_PATH = rote+'datasets'
    # 打开TIFF文件
    geotransform1, proj1, image1, width1, height1 = read_tiff((IN_PATH + '/{}/{}_wetland_image-1.tif').format(year, year))
    geotransform2, proj2, image2, width2, height2 = read_tiff((IN_PATH + '/{}/{}_wetland_image-2.tif').format(year, year))
    geotransform3, proj3, gt, width3, height3 = read_tiff((IN_PATH + '/{}/{}_label_tiff.tif').format(year, year))

    # 读取所有波段的数据
    data1 = np.array(image1)
    data2 = np.array(image2)
    label = np.array(gt)
    # 打印数据形状和类型
    print("Data1 Shape:", data1.shape)
    print("Data Type:", data1.dtype)
    print("Data2 Shape:", data2.shape)
    print("label Shape:", label.shape)


    #mat_data = sio.loadmat(IN_PATH + '/IN/Indian_pines_corrected.mat')
    data_IN = np.concatenate((data1 ,data2),axis=2)
    print(data_IN.max())
    print(data_IN.min())
    #mat_gt = sio.loadmat(IN_PATH + '/IN/Indian_pines_gt.mat')
    gt_IN = label
    print ("data_gt:",data_IN.shape)

    # Input dataset configuration to generate 103x7x7 HSI samples
    new_gt_IN = gt_IN

    #batch_size = 16
    nb_classes = 6
    #img_rows, img_cols =  7, 7 # 9, 9        

    INPUT_DIMENSION_CONV = 9
    INPUT_DIMENSION = 9

    # 20%:10%:70% data for training, validation and testing

    TOTAL_SIZE = np.count_nonzero(new_gt_IN)
    print("TOTAL_SIZE",TOTAL_SIZE)
    # VAL_SIZE = 4281

    TRAIN_SIZE = args.sample
    # DEV_SIZE = 200
    # VAL_SIZE = 400
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE # - DEV_SIZE - VAL_SIZE
    # VALIDATION_SPLIT = 0.9                      # 20% for trainnig and 80% for validation and testing

    sample = [np.sum(new_gt_IN==1),np.sum(new_gt_IN==2),np.sum(new_gt_IN==3),np.sum(new_gt_IN==4),np.sum(new_gt_IN==5),np.sum(new_gt_IN==6)]#,np.sum(new_gt_IN==7),np.sum(new_gt_IN==8)]
    print("sample number:",sample)
    sample_list=[x* TRAIN_SIZE/TOTAL_SIZE for x in sample]
    print("sample_list:",sample_list)
    sample_list_round = list(map(round, sample_list))
    print("sample_list_round:",sample_list_round)
    decimal = [sample_list[i]-sample_list_round[i] for i in range(0,len(sample_list))] #range后还可以加if条件筛选
    print("decimal:",decimal)
    
    #####判断相加是否为样本总数，如果不是则补充到与样本总数相同
    if sum(sample_list_round) > TRAIN_SIZE:
        min_value = min(decimal) # 求列表最小值
        min_idx = decimal.index(min_value) # 求最小值对应索引
        sample_list_round[min_idx] = sample_list_round[min_idx]-1
        SAMPLE = sample_list_round
    elif sum(sample_list_round) < TRAIN_SIZE:
        max_value = max(decimal) # 求列表最小值
        max_idx = decimal.index(max_value) # 求最小值对应索引
        sample_list_round[max_idx] = sample_list_round[max_idx]+1
        SAMPLE = sample_list_round
    else:
        #sample_list[1] = sample_list[1]-1
        SAMPLE = sample_list_round
    #sample_400 = [x* 2 for x in sample_200 ]
    print("sample",SAMPLE)

    img_channels = 9
    PATCH_LENGTH = 4                #Patch_size 9*9

    MAX = data_IN.max()
    #data_IN = np.transpose(data_IN, (2,0,1))


    data_IN = data_IN - np.mean(data_IN, axis=(1,2), keepdims=True)
    data_IN = data_IN / MAX
    print("after_norm max:",data_IN.max())
    print("after_norm min:",data_IN.min())
    data = data_IN.reshape(np.prod(data_IN.shape[:1]),np.prod(data_IN.shape[1:]))
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)

    whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])
    #whole_data = whole_data - np.mean(whole_data, axis=(1,2), keepdims=True)
    padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

    CATEGORY = 6

    train_data = np.zeros((TRAIN_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    test_data = np.zeros((TEST_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    all_data = np.zeros((TOTAL_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))

    # all_indices, train_indices, dev_indices, val_indices, test_indices = rsampling(gt)
    all_indices, train_indices, test_indices = rSampling(gt,SAMPLE)

    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1
    y_all = gt[all_indices] - 1

    train_assign = indexToAssignment(train_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, train_assign[i][0], train_assign[i][1])
        
    test_assign = indexToAssignment(test_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, test_assign[i][0], test_assign[i][1])
            
    all_assign = indexToAssignment(all_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for i in range(len(all_assign)):
        all_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_assign[i][0], all_assign[i][1])


    import torch
    from torch.utils import data

    df1 = pd.DataFrame(columns=['epoch','train Loss','training accuracy'])#列名
    df1.to_csv(args.save + '/train_loss_acc.csv',index=False) #路径可以根据需要更改
    df2 = pd.DataFrame(columns=['epoch','validation accuracy'])#列名
    df2.to_csv(args.save + '/val_acc.csv',index=False) #路径可以根据需要更改

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cuda', args.gpu)
    #torch.cudnn.benchmark = True

    # Parameters
    params = {'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 8}
    max_epochs = 200

    # Generators
    training_set = HSIDataset(range(len(train_indices)), train_data, y_train)
    training_generator = DataLoader(training_set, **params)

    validation_set = HSIDataset(range(len(test_indices)), test_data, y_test)
    validation_generator = DataLoader(validation_set, **params)

    all_set = HSIDataset(range(len(all_indices)), all_data, y_all)
    all_generator = DataLoader(all_set, **params)


    trainloader = torch.utils.data.DataLoader(training_set, batch_size=50, shuffle=True, num_workers=8)

    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=50, shuffle=False, num_workers=8)

    allloader = torch.utils.data.DataLoader(all_set, batch_size=50, shuffle=False, num_workers=8)

    batch_num = args.sample/args.batch_size
    print("batch_num:",batch_num)


    import torch
    import torch.optim as optim

    if args.model == 'Coatnet_aux':
        net = coatnet_7()
        trained_net = coatnet_8()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='3DCSN':
        net = HyperCLR(channel=15,output_units=6,windowSize=25)
        trained_net = HyperCLR(channel=15,output_units=6,windowSize=25)
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='ResNet18':
        net = resnet18()
        trained_net = resnet18()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='Vit':
        net = vit()
        trained_net = vit()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='Coatnet':
        net = coatnet_8()
        trained_net = coatnet_8()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='APSMnet_Drop_2aux':
        net = APSMnet_Drop1()
        trained_net = APSMnet_Drop1()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='APSMnet':
        net = APSMnet1()
        trained_net = APSMnet2()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='APSMnet_Drop':
        net = APSMnet_Drop1()
        trained_net = APSMnet_Drop2()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.model =='APSMnet_2aux':
        net = APSMnet1()
        trained_net = APSMnet1()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    
    else:
        logging.error("No such model in our zoo!")



    net.cuda()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.RMSprop(net.parameters())
    
    #optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    #optimizer = Lion(net.parameters(), lr=1e-4, weight_decay=1e-2)

    best_pred = 0
    SAVE_PATH3 = args.save + '/' + str(args.model) + '_sample' + str(args.sample) + '.pth' 
    #torch.save(net.state_dict(), SAVE_PATH)

    train_tic = time.time()
    print("It is Here1!!")
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("It is Here2!!")
        running_loss = 0.0
        total_loss = 0.0
        #iters = len(trainloader)
        net = net.train()
        tr_AC = 0
        total_label = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
         
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #outputs , aux_logits2, aux_logits1 = net(inputs.float())
            if (args.model == 'Coatnet_aux') or (args.model == 'APSMnet_Drop_2aux') or (args.model =='APSMnet') or (args.model =='APSMnet_Drop') or (args.model == 'APSMnet_2aux'):
                outputs , aux_logits2, aux_logits1 = net(inputs.float())
                loss0 = criterion(outputs, labels.long())
                loss1 = criterion(aux_logits1, labels.long())       #计算辅助分类器1与真实标签之间的损失
                loss2 = criterion(aux_logits2, labels.long())       #计算辅助分类器2与真实标签之间的损失
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3                    #将三个损失相加，0.3是因为在原论文中按0.3的权重
            else:
                outputs = net(inputs.float()) 
                loss = criterion(outputs, labels.long())
            #outputs = net(inputs.float())
            #loss = criterion(outputs, labels.long())
            _, predicted = torch.max(outputs.data, 1)
            total_label += labels.size(0)
            tr_AC += (predicted == labels.long()).sum().item()
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss +=loss.item()
            if i % 4 == 3:    # print every 2000 mini-batches
                logging.info('[%d, %5d] loss: %.4f' %
                    (epoch + 1, i + 1, running_loss / 4))
                running_loss = 0.0
        #schedular.step()
        t_loss = total_loss / batch_num
        train_loss = "%f"%t_loss
        print("total_label:",total_label)
        train_acc = tr_AC / total_label
        train_acc = "%g"%train_acc

        
        list = [epoch,train_loss,train_acc]
        data = pd.DataFrame([list])
        data.to_csv(args.save + '/train_loss_acc.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了


        if epoch % 10 == 0 or epoch > 190: 
            correct = 0
            total = 0
            net = net.eval()
            counter = 0 
            with torch.no_grad():
                for data in validationloader:
        #             if counter <= 10:
        #                 counter += 1
                    images, labels = data
                    images, labels = images.cuda(), labels.cuda()
                    outputs = net(images.float())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.long()).sum().item()

            new_pred = correct / total
            print("validation_total:",total)
            print("New_pred",new_pred)
            logging.info('Accuracy of the network on the validation set: %.5f %%' % (
                100 * new_pred))
           
            
            val_acc = "%g"%new_pred
            list = [epoch,val_acc]
            data = pd.DataFrame([list])
            data.to_csv(args.save + '/val_acc.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

            if new_pred > best_pred:
                logging.info('new_pred %f', new_pred)
                logging.info('best_pred %f', best_pred)
                torch.save(net.state_dict(), SAVE_PATH3)
                best_pred=new_pred
            
    logging.info('Finished Training')
    train_toc = time.time()
    print("Finished Training Time:",train_toc)
    logging.info('Training stage elapses:')

    logging.info(train_toc-train_tic)

    
    #trained_net = coatnet_8()

    val_tic = time.time()

    trained_net.load_state_dict(torch.load(SAVE_PATH3),strict=False)
    trained_net.eval()
    trained_net = trained_net.cuda()

    label_val = []
    pred_val = []

    with torch.no_grad():
        for data in validationloader:
            images, labels = data
            #label_val = torch.stack([label_val.type_as(labels), labels])
            label_val.append(labels)
            
            images, labels = images.cuda(), labels.cuda()
            outputs = trained_net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            #pred_val = torch.stack([pred_val.type_as(predicted), predicted])
            pred_val.append(predicted)
            
    label_val_cpu = [x.cpu() for x in label_val]
    pred_val_cpu = [x.cpu() for x in pred_val]

    label_cat = np.concatenate(label_val_cpu)
    pred_cat = np.concatenate(pred_val_cpu)

    matrix = metrics.confusion_matrix(label_cat, pred_cat)

    OA, AA_mean, Kappa, AA = cal_results(matrix)

    logging.info('OA, AA_Mean, Kappa: %f, %f, %f, ', OA, AA_mean, Kappa)
    logging.info(str(("AA for each class: ", AA)))
    val_toc = time.time()
    logging.info(str('Validation stage elapses:'))
    logging.info(val_toc-val_tic)
    logging.info('The number of each training sample:')
    logging.info(SAMPLE)
    logging.info('The total number of each sample:')
    logging.info(sample)


# # generate classification maps

# all_pred = []

# with torch.no_grad():
#     for data in allloader:
#         images, _ = data
#         images, _ = images.cuda(), labels.cuda()
#         outputs = trained_net(images.float())
#         _, predicted = torch.max(outputs.data, 1)
#         all_pred.append(predicted)

# all_pred = torch.cat(all_pred)
# all_pred = all_pred.cpu().numpy() + 1

# y_pred = predVisIN(all_indices, all_pred, 610, 340)


# #plt.plot(x, y)
# plt.imshow(y_pred)
# plt.axis('off')
# fig_path = './Cmaps/PC_' + str(args.model) + '.png'
# plt.savefig(fig_path, bbox_inches=0)
# #plt.savefig(fig_path, bbox_inches='tight')
