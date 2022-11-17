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

import torch.optim as optim
from torch.autograd import Variable
from torchvision import models

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import warnings
warnings.filterwarnings("ignore")
import random
# import cv2
from scipy.stats import spearmanr
use_gpu = True
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, flag, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=' ')
        self.root_dir = root_dir
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        try:
            if self.flag == 1:
                img_name = str(os.path.join(self.root_dir,(str(self.images_frame.iloc[idx, 0])+'.jpg')))
            else:
                img_name = str(os.path.join(self.root_dir, str(self.images_frame.iloc[idx, 0])))
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
        except Exception as e:
            pass



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
        im = image /1.0# / 255
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



class AesModel(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(AesModel, self).__init__()

        self.fc1_2 = nn.Linear(inputsize, 2048)
        self.bn1_2 = nn.BatchNorm1d(2048)
        self.drop_prob = (1 - keep_probability)
        self.relu1_2 = nn.PReLU()
        self.drop1_2 = nn.Dropout(self.drop_prob)
        self.fc2_2 = nn.Linear(2048, 1024)
        self.bn2_2 = nn.BatchNorm1d(1024)
        self.relu2_2 = nn.PReLU()
        self.drop2_2 = nn.Dropout(p=self.drop_prob)
        self.fc3_2 = nn.Linear(1024, num_classes)
        self.soft = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1_2(x)
        out_a = self.bn1_2(out_a)
        out_a = self.relu1_2(out_a)
        out_a = self.drop1_2(out_a)
        out_a = self.fc2_2(out_a)
        out_a = self.bn2_2(out_a)
        out_a = self.relu2_2(out_a)
        out_a = self.drop2_2(out_a)
        out_a = self.fc3_2(out_a)
        out_a = self.soft(out_a)
        # out_s = self.soft(out_a)
        # out_a = torch.cat((out_a, out_p), 1)


        # out_a = self.sig(out)
        return out_a

class PerModel(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(PerModel, self).__init__()

        self.fc1_1 = nn.Linear(inputsize, 2048)
        self.bn1_1 = nn.BatchNorm1d(2048)
        self.drop_prob = (1 - keep_probability)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(2048, 1024)
        self.bn2_1 = nn.BatchNorm1d(1024)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(1024, 5)
        self.bn3_1 = nn.BatchNorm1d(5)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

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

        return out_p

class convNet(nn.Module):
    #constructor
    def __init__(self,resnet,aesnet,pernet):
        super(convNet, self).__init__()
        #defining layers in convnet
        self.resnet=resnet
        self.AesNet=aesnet
        self.PerNet=pernet
    def forward(self, x):
        x=self.resnet(x)
        x1=self.AesNet(x)
        x2=self.PerNet(x)
        return x1, x2

def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        for batch_idx, data in enumerate(dataloader_valid):
            inputs = data['image']
            batch_size = inputs.size()[0]
            labels = data['rating'].view(batch_size, -1)[:,-1]
            labels= labels.view(batch_size, -1)
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs_a, outputs_p = model(inputs)
            sheet = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            sss = outputs_a.mul(Variable(sheet.cuda()))
            out = torch.sum(sss, 1).view(batch_size, -1)
            ratings.append(labels.float())
            predictions.append(out.float())

    ratings_i = np.vstack(ratings)
    predictions_i = np.vstack(predictions)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)
    return sp


def train_model(model, criterion, optimizer, lr_scheduler, dataloader_train, dataloader_train_2, dataloader_valid, num_epochs, best_spp):
    since = time.time()
    train_loss_average = []
    test_loss = []
    best_model = model
    best_loss = 100
    spp = 0
    criterion.cuda()
    model.cuda()
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            total_size = 0
            running_loss = 0.0
            if phase == 'train':
                mode = 'train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode

                dataloader_p = dataloader_train_2
                dataloader = dataloader_train

                counter = 0
                dataiter = iter(enumerate(dataloader_p))
                for batch_idx, data in enumerate(dataloader):
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    total_size = total_size + batch_size
                    labels = data['rating'].view(batch_size, -1)[:,:-1]

                    if use_gpu:
                        try:
                            inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                        except:
                            print(inputs, labels)
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    # wrap them in Variable

                    idx, data_p = next(dataiter)
                    if idx >= len(dataloader_p)-1:
                        dataiter = iter(enumerate(dataloader_p))
                    inputs_p = data_p['image']
                    batch_size1 = inputs_p.size()[0]
                    labels_p = data_p['rating'].view(batch_size1, -1)
                    if use_gpu:
                        try:
                            inputs_p, labels_p = Variable(inputs_p.float().cuda()), Variable(labels_p.float().cuda())
                        except:
                            print(inputs_p, labels_p)
                    else:
                        inputs_p, labels_p = Variable(inputs_p), Variable(labels_p)

                    optimizer.zero_grad()
                    outputs_s, outputs_p = model(inputs)
                    loss1 = criterion(outputs_s, labels)
                    loss1.backward()
                    optimizer.step()

                    optimizer.zero_grad()
                    outputs_ss, outputs_pp = model(inputs_p)
                    loss2 = criterion(outputs_pp, labels_p)
                    loss2.backward()
                    optimizer.step()

                    if counter % 20 == 0:
                        print("---Epoch is %d, Iteration is %d/%d, Loss is %f---" % (
                        epoch, counter, len(dataloader), loss1.data[0]))
                    counter += 1
                    try:
                        running_loss += loss1.data[0]
                    except:
                        print('unexpected error, could not calculate loss or do a sum.')
                # print('trying epoch loss')
                epoch_loss = running_loss / len(dataloader)
                print('average {} Loss: {:.4f} '.format(
                    phase, epoch_loss))
            else:
                model.eval()
                mode = 'val'
                dataloader = dataloader_valid

                counter = 0
                # Iterate over data.
                for batch_idx, data in enumerate(dataloader):
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    total_size = total_size + batch_size
                    labels = data['rating'].view(batch_size, -1)[:,:-1]
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
                    outputs_s, outputs_p = model(inputs)
                    loss = criterion(outputs_s, labels)

                    if counter % 20 == 0:
                        print("---Epoch is %d, Iteration is %d/%d, Loss is %f----" % (
                        epoch, counter, len(dataloader), loss.data[0]))
                    counter += 1
                    try:
                        running_loss += loss.data[0]
                    except:
                        print('unexpected error, could not calculate loss or do a sum.')

                # print('trying epoch loss')
                epoch_loss = running_loss / len(dataloader)
                print('average {} Loss: {:.4f} '.format(
                    phase, epoch_loss))
                # deep copy the model
                if phase == 'val':
                    test_loss.append(epoch_loss)
                    spp = computeSpearman(dataloader, model)[0]
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        print('new best loss = %f' % epoch_loss)
                    if spp > best_spp:
                        best_spp = spp
                        best_model = copy.deepcopy(model)
                        print('new best srocc = %f' % spp)
                    print('best loss: %f, best srocc: %f' % (epoch_loss, best_spp))
    # Save model
    torch.save(model.cuda(),
                'Personality_Model/AVA_Personality_normalized_inception_DB.pt')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('returning and looping back')
    return best_model.cuda(), train_loss_average, test_loss, spp



def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.5**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

def calc_auto(num, channels):
    lst = [1, 2, 4, 8, 16, 32]
    return sum(map(lambda x: x ** 2, lst[:num])) * channels


model_path = 'Personality_Model'
if not os.path.exists(model_path):
    os.mkdir(model_path)
personality_data_path = '/home/hancheng/Image_dataset/PsychoFlickr_Project/data'
personality_image_path = '/home/hancheng/Image_dataset/PsychoFlickr_Project/PsychoFlickr'
AVA_data_path = '/home/hancheng/Image_dataset/AVA_Project/data'
AVA_image_path = '/home/hancheng/Image_dataset/AVA_Project/AVA_image'

transformed_dataset_train = ImageRatingsDataset(csv_file= os.path.join(AVA_data_path, 'Train_distrubtion.txt'),
                                                root_dir= AVA_image_path,
                                                flag=1,
                                           transform=transforms.Compose([ Rescale(output_size=(350,350)),
                                                                        RandomCrop(output_size=(299,299)),
                                                                          Normalize(),
                                                                          ToTensor()]))
transformed_dataset_train_2 = ImageRatingsDataset(csv_file=os.path.join(personality_data_path, 'Personality.csv'),
                                                root_dir= personality_image_path,
                                                  flag=0,
                                           transform=transforms.Compose([ Rescale(output_size=(350,350)),
                                                                        RandomCrop(output_size=(299,299)),
                                                                          Normalize(),
                                                                          ToTensor()]))
transformed_dataset_valid = ImageRatingsDataset(csv_file=os.path.join(AVA_data_path, 'Test_distrubtion.txt'),
                                                root_dir= AVA_image_path,
                                                flag=1,
                                           transform=transforms.Compose([Rescale(output_size=(299,299)),
                                                                         Normalize(),
                                                                        ToTensor()]))

bsize=50

dataloader_train = DataLoader(transformed_dataset_train, batch_size=bsize,
                        shuffle=True, num_workers=0,collate_fn=my_collate)
dataloader_train_2 = DataLoader(transformed_dataset_train_2, batch_size=bsize,
                        shuffle=True, num_workers=0,collate_fn=my_collate)
dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=int(bsize),
                        shuffle=True, num_workers=0,collate_fn=my_collate)


best_spp = -1
num_epochs = 20
for i in range(20):
    # model_ft = models.densenet121(pretrained=True)
    model_ft = models.inception_v3(pretrained=True)
    model_ft.aux_logits = False
    num_ftrs = model_ft.fc.out_features
    net1 = AesModel(10, 0.5, num_ftrs)
    net2 = PerModel(1, 0.5, num_ftrs)
    net3 = convNet(resnet=model_ft, aesnet=net1, pernet=net2)
    # net_2 = (torch.load('Personality_Model/FlickrAES_Personality_normalized_resnet18.pt'))
    model_ft = net3

    #device = cuda
    criterion = nn.MSELoss()
    criterion.cuda()
    model_ft.cuda()

    ignored_params = list(map(id, net1.parameters() or net2.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model_ft.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': net1.parameters() or net2.parameters(), 'lr': 1e-3}
    ], lr=1e-5)  # , weight_decay=0.00001)


    print('repeat: %d, best srocc: %f' %(i, best_spp))
    model_s, train_loss, test_loss, best_spp = train_model(model_ft, criterion, optimizer, exp_lr_scheduler,
                                                                dataloader_train, dataloader_train_2,
                                                                dataloader_valid, num_epochs, best_spp)