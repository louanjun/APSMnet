
import numpy as np
import pandas as pd
import torch
from torch import nn
from osgeo import gdal
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from coatnet_aux import coatnet_8
#from APSMnet import APSMnet1,APSMnet2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils import data
from torch.utils.data import DataLoader
from utils import cal_results, predVisIN
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from APSMnet_Drop import APSMnet_Drop1,APSMnet_Drop2
from ResNet import resnet18
from vit import vit
from DCSN import HyperCLR

class TIFFDataset(Dataset):
    def __init__(self, tiff_image):
        super(TIFFDataset, self).__init__()
        tiff_image = tiff_image[:, :, :]
       # print("tiff_image0",tiff_image.shape)
        #tiff_image = tiff_image.transpose(1, 2, 0)
        #self.image = tiff_image.reshape(-1, 9)
        self.image = tiff_image

    def __getitem__(self, i):
        return self.image[i]

    def __len__(self):
        return self.image.shape[0]


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

def rSampling(groundTruth, sample_num ):              #divide dataset into train and test datasets
    

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
    
def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix

class HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples):
        
        self.list_IDs = list_IDs
        self.samples = samples

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        return X
    
def read_tiff(tiff_path):
    dataset = gdal.Open(tiff_path)
    height, width = dataset.RasterYSize, dataset.RasterXSize
    geotransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    image = dataset.ReadAsArray()
    image[np.isnan(image)] = 0  # 替换nan为0
    del dataset
    return geotransform, proj, image, width, height

def write_tiff(image, save_path, geotrans, project, width, height):
    datatype = gdal.GDT_Byte
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, width, height, 1, datatype)
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(project)
    dataset.GetRasterBand(1).WriteArray(image)
    del dataset


if __name__ == '__main__':
    
    #  添加下面这行代码
    #import torch.multiprocessing
    #torch.multiprocessing.freeze_support()
    rote = 'E:/GitHub/APSMnet/'
    IN_PATH = rote+'datasets'
    year = 2000
   
    # 打开TIFF文件
    model_name = 'APSMnet_Drop_2aux'   #  'ResNet18'  'Coatnet'  'Vit' '3DCSN' 'APSMnet_Drop_2aux'  
    timeOfModel = '20230723-134312'
    sample_num = 4000
    learning_rate = 0.002
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
    '''
    geotransform2, proj2, image2, width2, height2 = read_tiff(IN_PATH + '/2000_part/2000_wetland_image.tif')
    geotransform3, proj3, gt, width3, height3 = read_tiff(IN_PATH + '/2000_part/2000_label_tiff.tif')
    label = np.array(gt)
    data2 = np.array(image2)
    data_IN = data2
    '''
    new_gt_IN = label
    #mat_gt = sio.loadmat(IN_PATH + '/IN/Indian_pines_gt.mat')
    INPUT_DIMENSION_CONV = 9
    INPUT_DIMENSION = 9

    # 20%:10%:70% data for training, validation and testing

    TOTAL_SIZE = np.count_nonzero(data_IN[1,:,:])
    #cnt_array0 = np.where(data_IN[0,:,:],0,1)
    #print("zero_num",np.sum(cnt_array0))

    print("non_zero",np.count_nonzero(data_IN[0,:,:]),np.count_nonzero(data_IN[1,:,:]),np.count_nonzero(data_IN[2,:,:]),
                        np.count_nonzero(data_IN[3,:,:]),np.count_nonzero(data_IN[4,:,:]),np.count_nonzero(data_IN[5,:,:]),
                        np.count_nonzero(data_IN[6,:,:]),np.count_nonzero(data_IN[7,:,:]),np.count_nonzero(data_IN[8,:,:]))  #除了第6、7波段之外其他波段原图像没有0值，可以以其他波段非0值作为索引
    # 40076552

    

    img_channels = 9  

    PATCH_LENGTH = 4                #Patch_size 9*9
    
    print("before_norm max:", data_IN.max())
    print("before_norm min:", data_IN.min())   
    print("data_In:",data_IN.shape)
    
    MAX = data_IN.max() 
    print("Max:",MAX)
    mask = data_IN != 0  # 创建一个布尔掩码，标记非零元素位置
    mean_values = np.mean(data_IN, axis=(1, 2), keepdims=True)  # 计算非零元素在 (1, 2) 轴上的均值
    mean_values = np.where(mask, mean_values, 0)  # 非零位置保留均值，零位置设为0

    data_IN = data_IN - mean_values
    data_IN = data_IN / MAX
    print("after_norm max:",data_IN.max())
    print("after_norm min:",data_IN.min())
    

    data = data_IN.reshape(np.prod(data_IN.shape[:1]),np.prod(data_IN.shape[1:]))  #9,10877*15086
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)  #10877*15086,
    print("data:",data.shape)
    print("gt:",gt.shape)
    # generate classification maps
    print("max label:", gt.max())
    print("min label:", gt.min())
   
    whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2]) #(9, 10877, 15084)
    print("whole_data:",whole_data.shape)
    #whole_data = whole_data - np.mean(whole_data, axis=(1,2), keepdims=True)
    padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)#(9, 10885, 15092)
    print("padded_data:",padded_data.shape)
    Batch = 80000   #500000
    num_batches = TOTAL_SIZE // Batch 
    print("num_batches:",num_batches)

    all_indices = [j for j, x in enumerate(whole_data[0,:,:].ravel().tolist()) if x != 0]  #寻找第0个波段不为0元素的索引
    all_assign = indexToAssignment(all_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    print("all_indices_shape:",len(all_indices)) # 40076552
    print("all_assign_shape:",len(all_assign))

    del whole_data
    
    
    all_data = np.zeros((Batch, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    #print("all_data:",all_data.shape)
    

    if model_name =='Coatnet':
        trained_net = coatnet_8().to(device)
    elif model_name =='APSMnet_Drop_2aux':
        trained_net = APSMnet_Drop1().to(device)
    elif model_name =='Vit':
        trained_net = vit().to(device)
    elif model_name =='ResNet18':
        trained_net = resnet18().to(device)
    elif model_name =='3DCSN':
        trained_net = HyperCLR(channel=15,output_units=6,windowSize=25).to(device)
    else:
        print("No such model in our zoo!")
    
    ############################################除了最后一个batch
    all_pred = []
    for n in range(num_batches):
        print("n",n)
        for i in range(Batch):
            #if i+n*Batch < len(all_indices):
                all_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_assign[i+n*Batch][0], all_assign[i+n*Batch][1])
       
        # Parameters
        params = {'batch_size': 50,
            'shuffle': True,
            'num_workers': 2}
        max_epochs = 200 

        all_set = HSIDataset(range(Batch), all_data)
        print("all_data:",all_data.shape)
        all_generator = DataLoader(all_set, **params)
        allloader = torch.utils.data.DataLoader(all_set, batch_size=50, shuffle=False, num_workers=2)

        load_path = (rote+'train_model/{}-train-sample{}-model-{}-{}-lr{}/{}_sample{}.pth').format(year, sample_num, model_name, timeOfModel, learning_rate, model_name, sample_num)
        
       # trained_net = coatnet_8().to(device)
        trained_net.load_state_dict(torch.load(load_path),strict=False)
        trained_net.eval() 

    
        with torch.no_grad():
            for data in allloader:
                images = data
                #print("images:",images.shape)
                images = images.to(device)
                outputs = trained_net(images.float())
                _, predicted = torch.max(outputs.data, 1)
                all_pred.append(predicted)
    all_pred = torch.cat(all_pred)
    print("all_pred_size_withoutres:",all_pred.shape)
    ############################################最后一个batch
    
    Batch_res = TOTAL_SIZE % Batch
    res_data = np.zeros((Batch_res, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
   
    for i in range(Batch_res):
        res_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_assign[num_batches*Batch+i][0], all_assign[num_batches*Batch+i][1])
        #print("index:",num_batches*Batch+i)
     # Parameters
    params = {'batch_size': 50,
        'shuffle': True,
        'num_workers': 3}
    max_epochs = 200 

    res_set = HSIDataset(range(Batch_res), res_data)
    print("res_data:",res_data.shape)
    res_generator = DataLoader(res_set, **params)
    resloader = torch.utils.data.DataLoader(res_set, batch_size=50, shuffle=False, num_workers=3)

    
    #trained_net = coatnet_8().to(device)
   
    trained_net.load_state_dict(torch.load(load_path),strict=False)
    trained_net.eval() 
    
    
    res_pred = []
    with torch.no_grad():
        for data in resloader:
            images = data
            #print("images:",images.shape)
            images = images.to(device)
            outputs = trained_net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            res_pred.append(predicted)
   
    res_pred = torch.cat(res_pred)
  
   #########################################################
    #绘图
    
    all_pred = torch.cat((all_pred,res_pred))
    all_pred = all_pred.cpu().numpy() + 1
    print("all_pred_size:",all_pred.shape)
    if all_pred.ndim > 1:
        all_pred = np.ravel(all_pred)
    
    out_image = np.zeros( width3*height3)
    out_image[all_indices] = all_pred

    out_image = out_image.reshape((height3, width3))
    #all_pred = torch.cat(all_pred)

    #all_pred = all_pred.cpu().numpy() + 1

    out_tiff_path = (rote+'result_image/{}_classify_image{}_sample{}_{}.tif').format(timeOfModel, year, sample_num, model_name)
    write_tiff(out_image, out_tiff_path, geotransform3, proj3, width3, height3)

