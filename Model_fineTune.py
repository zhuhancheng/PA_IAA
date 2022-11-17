from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1:]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}

class BaselineModel(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel, self).__init__()
        self.fc1_1 = nn.Linear(inputsize, 1024)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(1024, 512)
        self.bn2_1 = nn.BatchNorm1d(512)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(512, 5)
        self.bn3_1 = nn.BatchNorm1d(5)
        self.tanh = nn.Tanh()

        self.fc1_2 = nn.Linear(inputsize, 1024)
        self.bn1_2 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1_2 = nn.PReLU()
        self.drop1_2 = nn.Dropout(self.drop_prob)
        self.fc2_2 = nn.Linear(1024, 512)
        self.bn2_2 = nn.BatchNorm1d(512)
        self.relu2_2 = nn.PReLU()
        self.drop2_2 = nn.Dropout(p=self.drop_prob)
        self.fc3_2 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out_p = self.fc1_1(x)
        out_p = self.bn1_1(out_p)
        out_p = self.relu1_1(out_p)
        out_p = self.drop1_1(out_p)
        out_p = self.fc2_1(out_p)
        out_p = self.bn2_1(out_p)
        out_p = self.relu2_1(out_p)
        out_p = self.drop2_1(out_p)
        out_p = self.fc3_1(out_p)
        out_p = self.bn3_1(out_p)
        out_p = self.tanh(out_p)

        out_a = self.fc1_2(x)
        out_a = self.bn1_2(out_a)
        out_a = self.relu1_2(out_a)
        out_a = self.drop1_2(out_a)
        out_a = self.fc2_2(out_a)
        out_a = self.bn2_2(out_a)
        out_a = self.relu2_2(out_a)
        out_a = self.drop2_2(out_a)
        out_a = self.fc3_2(out_a)
        out_a = self.sig(out_a)
        # out_a = torch.cat((out_a, out_p), 1)


        # out_a = self.sig(out)
        return out_a, out_p

class convNet(nn.Module):
    #constructor
    def __init__(self,resnet,mynet):
        super(convNet, self).__init__()
        #defining layers in convnet
        self.resnet=resnet
        self.myNet=mynet
    def forward(self, x):
        x=self.resnet(x)
        x=self.myNet(x)
        return x

class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.tanh(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out

class Net(nn.Module):
    def __init__(self , model, net):
        super(Net, self).__init__()
        self.resnet_layer = model.resnet
        self.net2 = model.myNet
        self.net3 = net


    def forward(self, x):
        x = self.resnet_layer(x)
        x1, x2 = self.net2(x)
        x3 = self.net3(x)
        x4 = torch.mul(x2,x3)
        x5 = torch.sum(x4, 1).view(-1,1)
        x = x1 + x5
        return x


def computeSpearman(dataloader_valid, model, net_2, epoch):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        for batch_idx, data in enumerate(dataloader_valid):
            inputs = data['image']
            batch_size = inputs.size()[0]
            labels = data['rating'].view(batch_size, -1)
            labels = labels / 5.0
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            if epoch == 0:
                outputs_a, ss = net_2(inputs)
            else:
                outputs_a = model(inputs)
            ratings.append(labels.float())
            predictions.append(outputs_a.float())

    ratings_i = np.vstack(ratings)
    predictions_i = np.vstack(predictions)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)
    return sp

def train_model(model, criterion, optimizer, dataloader_train, dataloader_valid, num_epochs=100):
    since = time.time()
    train_loss_average = []
    test_loss = []
    best_model = model
    best_loss = 100
    best_spearman = 0
    sp = 0
    spearman_test = 0
    net_2 = (torch.load(
        'Personality_Model/FlickrAES_Personality_normalized.pt'))
    criterion.cuda()
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in [ 'val','train']:
            if phase == 'train':
                mode = 'train'
                model.train()  # Set model to training mode
                dataloader = dataloader_train
            elif phase == 'val' and epoch == 0:
                net_2.eval()
                mode = 'val'
                dataloader = dataloader_valid
            else:
                model.eval()
                mode = 'val'
                dataloader = dataloader_valid

            running_loss = 0.0
            model.cuda()

            counter = 0
            # Iterate over data.
            for batch_idx, data in enumerate(dataloader):
                inputs = data['image']
                batch_size = inputs.size()[0]
                labels = data['rating'].view(batch_size, -1)
                labels = labels / 5.0
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    except:
                        print(inputs, labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # wrap them in Variable

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                if phase == 'val' and epoch == 0:
                    outputs, ss = net_2(inputs)
                else:
                    outputs = model(inputs)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                try:
                    running_loss += loss.data[0]
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            # print('trying epoch loss')
            epoch_loss = running_loss / len(dataloader)
            # print('{} Loss: {:.4f} '.format(
            #    phase, epoch_loss))
            if phase == 'train':
                train_loss_average.append(epoch_loss)

            # deep copy the model
            if phase == 'val':
                test_loss.append(epoch_loss)
                sp = computeSpearman(dataloader, model, net_2, epoch)[0]
                if epoch == 0:
                    print('no train srocc: %4f' %sp)
                    spearman_test = sp

                if epoch_loss < best_loss:
                    best_loss = epoch_loss

                if sp > best_spearman:
                    best_spearman = sp

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f} best spearman: {:4f}'.format(best_loss, best_spearman))
    # print('returning and looping back')
    return model.cuda(), train_loss_average, np.asarray(test_loss), spearman_test, sp

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

data_dir = os.path.join('/home/hancheng/Image_dataset/Flickr_AES_Project/data')

workers_test = pd.read_csv(os.path.join(data_dir,  'test_workers_dataset_normalized.csv'), sep=' ')
worker_test_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_each_worker.csv'), sep=',')
generic_score = pd.read_csv(os.path.join(data_dir, 'image_average_worker_scores.csv'), sep=' ')

workers_fold = "Workers/"
if not os.path.exists(workers_fold):
    os.makedirs(workers_fold)

epochs = 20
spp = []
spt =[]
for worker_idx in range(37):

    worker = workers_test['worker'].unique()[worker_idx]
    print("----worker number: %2d---- %s" %(worker_idx, worker))
    num_images = worker_test_orignal[worker_test_orignal['worker'].isin([worker])].shape[0]
    print('train image: 100, test image: %d' %(num_images - 100))
    percent = 100 / num_images
    images = worker_test_orignal[worker_test_orignal['worker'].isin([worker])][[' imagePair', ' score']]

    srocc_list =[]
    no_train_srocc_list = []
    for i in range(0, 10):

        train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
        train_path = workers_fold + "train_scores_" + worker + ".csv"
        test_path = workers_fold + "test_scores_" + worker + ".csv"
        train_dataframe.to_csv(train_path, sep=',', index=False)
        valid_dataframe.to_csv(test_path, sep=',', index=False)

        output_size = (224, 224)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir='/home/hancheng/Image_dataset/Flickr_AES_Project/Images/',
                                                        transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                      RandomHorizontalFlip(0.5),
                                                                                      RandomCrop(
                                                                                          output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir='/home/hancheng/Image_dataset/Flickr_AES_Project/Images/',
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        bsize = 100

        dataloader_train = DataLoader(transformed_dataset_train, batch_size=bsize,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=50,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

        net2 = torch.load('Personality_Model/FlickrAES_Personality_normalized.pt')
        num_ftrs = net2.resnet.fc.out_features
        net3 = BaselineModel1(5, 0.5, num_ftrs)
        model_ft = Net(model=net2, net=net3)
        criterion = nn.MSELoss()
        ignored_params = list(map(id, model_ft.net3.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model_ft.parameters())
        optimizer = optim.Adam([
            {'params': base_params},
            {'params': model_ft.net3.parameters(), 'lr': 1e-3}
        ], lr=1e-5)

        model_s, train_loss, test_loss, no_train_sp, spearman = train_model(model_ft, criterion, optimizer, dataloader_train,
                                                               dataloader_valid, num_epochs=epochs)
        srocc_list.append(spearman)
        no_train_srocc_list.append(no_train_sp)
    srocc = np.mean(srocc_list)
    no_train_srocc = np.mean(no_train_srocc_list)
    print("-------average srocc is: %4f-------" % srocc)
    spp.append(srocc)
    spt.append(no_train_srocc)
sp_file = pd.DataFrame({'no_train': spt,
                       'train': spp})
sp_file.to_csv(os.path.join('results', 'Each_worker_SP_Flickr_AES.csv'), sep=',', index=False)
spear = np.mean(spp)
print(spear)