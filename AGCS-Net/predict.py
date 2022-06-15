import torch
from torchvision import transforms
from PIL import Image, ImageOps

import numpy as np
import scipy.misc as misc
import os
import glob

from utils.misc import thresh_OTSU, ReScaleSize, Crop
from utils.model_eval import eval

DATABASE = './octa/'
#
args = {
    'root'     : './dataset/' + DATABASE,               #'root'     : './dataset/octa/'
    'test_path': './dataset/' + DATABASE + 'test/',     #'test_path'     : './dataset/octa/test/'
    'pred_path': 'assets/' + 'octa/',                   #'pred_path':'assets/octa/'
    'img_size' : 512
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def rescale(img):
    w, h = img.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    img = img.crop(box)
    return img

def load_octa():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.tif')):  #'test_path'     : './dataset/octa/test/'
        basename = os.path.basename(file)
        file_name = basename[:-4]
        # print(file_name)
        image_name = os.path.join(args['test_path'], 'images', basename)
        # label_name = os.path.join(args['test_path'], 'label', file_name + '_nerve_ann.tif')
        label_name = os.path.join(args['test_path'], 'label', file_name + '.tif')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_net():
    net = torch.load('./checkpoint/U_Net_octa_200.pkl')#CS_Net_DRIVE_1000.pkl改为CS_Net_octa_600.pkl
    return net


def save_prediction(pred, filename=''):
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    mask = pred.data.cpu().numpy() * 255
    print(mask.shape)
    mask = np.transpose(np.squeeze(mask, axis=0), [1, 2, 0])
    print(mask.shape)
    mask = np.squeeze(mask, axis=-1)
    print(mask.shape)
    misc.imsave(save_path + filename + '.png', mask)


def predict():
    net = load_net()
    # images, labels = load_nerve()
    # images, labels = load_drive()
    # images, labels = load_stare()
    # images, labels = load_padova1()
    images, labels = load_octa()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    with torch.no_grad():
        net.eval()
        for i in range(len(images)):
            print(images[i])
            name_list = images[i].split('/')
            index = name_list[-1][:-4]
            image = Image.open(images[i])
            # image=image.convert("RGB")
            label = Image.open(labels[i])
            # image, label = center_crop(image, label)

            # for other retinal vessel
            # image = rescale(image)
            # label = rescale(label)
            # image = ReScaleSize_STARE(image, re_size=args['img_size'])
            # label = ReScaleSize_DRIVE(label, re_size=args['img_size'])

            # for OCTA
            # misc.imsave('output/'+str(index) + '_raw.png', image)
            # image = Crop(image)
            # misc.imsave('output/'+str(index) + '_crop.png', image)
            image = ReScaleSize(image)
            # misc.imsave('output/'+str(index) + '_resize.png', image)
            # label = Crop(label)
            label = ReScaleSize(label)
            # misc.imsave('output/'+str(index) + '.png', image)
            # misc.imsave(str(index) + '_pred.png', label)
            # print(label)
            # label.save('output/'+str(index) + '_pred.png')
            # label = label.resize((args['img_size'], args['img_size']))
            # if cuda
            image = transform(image).cuda()
            # image = transform(image)
            image = image.unsqueeze(0)
            output = net(image)

            save_prediction(output, filename=index + '_pred')
            # save_prediction(label, filename=index)
            save_path = args['pred_path'] + 'pred/'
            label.save(args['pred_path']+'label/'+str(index) + '.png')
            print("output saving successfully")


if __name__ == '__main__':
    predict()
    thresh_OTSU(args['pred_path'] + 'pred/')
