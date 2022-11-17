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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
            # rating = self.images_frame.iloc[idx, 1:]
            sample = {'image': image}

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
        image = sample['image']
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

        return {'image': image}


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
        image = sample['image']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        im = image /1.0# / 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double()}



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

def model_eval(model, dataloader_valid, root_dir, csv_file):

    im = []
    score = []
    distribute = []
    total_size = 0
    model.eval()
    model.cuda()
    dataloader = dataloader_valid
    counter = 0
    # Iterate over data.
    images_frame = pd.read_csv(csv_file, sep=' ')
    for batch_idx, data in enumerate(dataloader):
        inputs = data['image']
        batch_size = inputs.size()[0]
        total_size = total_size + batch_size
        img_name = str(os.path.join(root_dir, (str(images_frame.iloc[counter, 0]))))
        print('image name: {:s}'.format(img_name))
        im.append(str(images_frame.iloc[counter, 0]))
        counter += 1
        if use_gpu:
            try:
                inputs = Variable(inputs.float().cuda())
            except:
                print(inputs)
        else:
            inputs = Variable(inputs)
        # wrap them in Variable

        # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
        with torch.no_grad():
            outputs_a, outputs_p = model(inputs)
        sheet = torch.Tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        sss = outputs_a.mul(Variable(sheet.cuda()))
        out = torch.sum(sss, 1).view(batch_size, -1)
        a = out.view(-1).cpu().detach().numpy()
        b = round(a[0]/10, 4)
        score.append(b)
        distribute.append(np.array(outputs_a.view(-1).cpu().detach().numpy()))
    test_file = pd.DataFrame({'image_name': im,
                              # '1': distribute,
                           'score': score})
    test_file.round({'score': 4})
    dis_file = pd.DataFrame(distribute)
    test_file.to_csv(os.path.join(data_path, 'Our_results.txt'), sep=' ', index=False)
    test_file = pd.concat([test_file, dis_file], axis=1)
    test_file.to_csv(os.path.join(data_path, 'Results_Aes.csv'), sep=',', index=False)

def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

def calc_auto(num, channels):
    lst = [1, 2, 4, 8, 16, 32]
    return sum(map(lambda x: x ** 2, lst[:num])) * channels

data_path = '/home/hancheng/Aesthetic_test'
image_path = '/home/hancheng/Aesthetic_test/img'

transformed_dataset_valid = ImageRatingsDataset(csv_file=os.path.join(data_path, '1.txt'),
                                                root_dir= image_path,
                                                flag=0,
                                                transform=transforms.Compose([Rescale(output_size=(299,299)),
                                                                         Normalize(),
                                                                        ToTensor()]))
bsize=1
dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=int(bsize),
                        shuffle=False, num_workers=0,collate_fn=my_collate)


model_ft=torch.load('Personality_Model/Personality_normalized_inception.pt')
model_eval(model_ft, dataloader_valid, root_dir= data_path, csv_file=os.path.join(data_path, '1.txt'))
