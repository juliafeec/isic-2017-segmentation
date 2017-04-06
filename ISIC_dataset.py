import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import yaml
from sklearn.cross_validation import train_test_split
import pandas as pd
np.random.seed(4)

mean_imagenet = [123.68, 103.939, 116.779] # rgb

def get_labels(image_list, csv_file):
    image_list = [filename.split('.')[0] for filename in image_list]
    return pd.read_csv(csv_file,index_col=0).loc[image_list]['melanoma'].values.flatten().astype(np.uint8)

def get_mask(image_name, mask_folder, rescale_mask=True):
    img_mask = cv2.imread(os.path.join(mask_folder, image_name.replace(".jpg","_segmentation.png")), 
                              cv2.IMREAD_GRAYSCALE)
    if img_mask is None:
        img_mask = cv2.imread(os.path.join(mask_folder, image_name.replace(".jpg",".png")), 
                              cv2.IMREAD_GRAYSCALE)
    _,img_mask = cv2.threshold(img_mask,127,255,cv2.THRESH_BINARY)
    if rescale_mask:
        img_mask = img_mask/255.
    return img_mask
    
def get_color_image(image_name, image_folder, remove_mean_imagenet=True, use_hsv=False, remove_mean_samplewise=False):
    if remove_mean_imagenet and remove_mean_samplewise:
        raise Exception("Can't use both sample mean and Imagenet mean")
    img = cv2.imread(os.path.join(image_folder, image_name.replace(".jpg",".png")))
    if img is None:
        img = cv2.imread(os.path.join(image_folder, image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if remove_mean_imagenet:
        for channel in [0,1,2]:
            img[:,:,channel] -= mean_imagenet[channel]
    elif remove_mean_samplewise:
        img_channel_axis = 2
        img -= np.mean(img, axis=img_channel_axis, keepdims=True)
    if use_hsv:
        img_all = np.zeros((img.shape[0],img.shape[1],6))
        img_all[:,:,0:3] = img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_all[:,:,3:] = img_hsv
        img = img_all
        
    img = img.transpose((2,0,1)).astype(np.float32)
    return img
        
def load_images(images_list, height, width, image_folder, mask_folder, remove_mean_imagenet=True, rescale_mask=True, use_hsv=False, remove_mean_samplewise=False):
    if use_hsv:
        n_chan = 6
    else:
        n_chan = 3
    img_array = np.zeros((len(images_list), n_chan, height, width), dtype=np.float32)
    if mask_folder:
        img_mask_array = np.zeros((len(images_list), height, width), dtype=np.float32)
    i = 0
    for image_name in images_list:
        img = get_color_image(image_name, image_folder, remove_mean_imagenet=remove_mean_imagenet,use_hsv=use_hsv,remove_mean_samplewise=remove_mean_samplewise)
        img_array[i] = img
        if mask_folder:
            img_mask = get_mask(image_name, mask_folder, rescale_mask)
            img_mask_array[i] =img_mask
        i = i+1
    if not mask_folder:
        return img_array
    else:
        return (img_array, img_mask_array.astype(np.uint8).reshape((img_mask_array.shape[0],1,img_mask_array.shape[1],img_mask_array.shape[2])))

def train_test_from_yaml(yaml_file, csv_file):
    with open(yaml_file,"r") as f:
        folds = yaml.load(f); 
    train_list, test_list = folds["Fold_1"]
    train_label = get_labels(train_list, csv_file=csv_file)
    test_label = get_labels(test_list, csv_file=csv_file)
    return train_list, train_label, test_list, test_label

def train_val_split(train_list, train_labels, seed, val_split = 0.20):
    train_list, val_list, train_label, val_label = train_test_split(train_list, train_labels, test_size=val_split, stratify=train_labels, random_state=seed)
    return train_list, val_list, train_label, val_label

def train_val_test_from_txt(train_txt, val_txt, test_txt):
    train_list =[]; val_list = []; test_list = [];
    with open(train_txt) as t:
        for img in t:
            img = img.strip()
            if img.endswith(".jpg"):
                train_list.append(img)
    with open(val_txt) as t:
        for img in t:
            img = img.strip()
            if img.endswith(".jpg"):
                val_list.append(img)
    with open(test_txt) as t:
        for img in t:
            img = img.strip()
            if img.endswith(".jpg"):
                test_list.append(img)
    print "Found train: {}, val: {}, test: {}.".format(len(train_list),len(val_list),len(test_list))
    return train_list, val_list, test_list
    
def list_from_folder(image_folder):
    image_list = []
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".jpg"):
            image_list.append(image_filename)
    print "Found {} ISIC validation images.".format(len(image_list))
    return image_list

def move_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height=None, width=None, same_name=False):
    base_output_folder = output_image_folder
    base_output_mask_folder = output_mask_folder
    for k in range(len(images_list)):
        image_filename = images_list[k]
        image_name = os.path.basename(image_filename).split('.')[0]
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)
        if input_mask_folder and not os.path.exists(output_mask_folder):
            os.makedirs(output_mask_folder)
        if height and width:
            img = cv2.imread(os.path.join(input_image_folder,image_filename))
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_image_folder,image_name+".png"), img)
            if input_mask_folder:
                img_mask = get_mask(image_filename, input_mask_folder, rescale_mask=False)
                img_mask = cv2.resize(img_mask, (width, height), interpolation = cv2.INTER_CUBIC)
                _,img_mask = cv2.threshold(img_mask,127,255,cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(output_mask_folder,image_name+".png"), img_mask)
        else:
            if not same_name:
                shutil.copyfile(os.path.join(input_image_folder, image_filename), os.path.join(output_image_folder,image_name+".jpg"))
            else:
                img = cv2.imread(os.path.join(input_image_folder,image_filename))
                cv2.imwrite(os.path.join(output_image_folder,image_name+".png"), img)
            
            if input_mask_folder:
                image_mask_filename = image_filename.replace(".jpg","_segmentation.png")
                shutil.copyfile(os.path.join(input_mask_folder,image_mask_filename), os.path.join(output_mask_folder,image_name+".png"))
            
def resize_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height, width):
    return move_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height, width)

def get_mask_full_sized(mask_pred, original_shape, output_folder = None, image_name = None):
    mask_pred = cv2.resize(mask_pred, (original_shape[1], original_shape[0])) # resize to original mask size
    _,mask_pred = cv2.threshold(mask_pred,127,255,cv2.THRESH_BINARY)
    if output_folder and image_name:
        cv2.imwrite(os.path.join(output_folder,image_name.split('.')[0]+"_segmentation.png"), mask_pred)
    return mask_pred

def show_images_full_sized(image_list, img_mask_pred_array, image_folder, mask_folder, index, output_folder=None, plot=True):
    image_name = image_list[index]
    img = get_color_image(image_name, image_folder, remove_mean_imagenet=False).astype(np.uint8)
    img = img.transpose(1,2,0)
    if mask_folder:
        mask_true = get_mask(image_name, mask_folder, rescale_mask=False)
    mask_pred = get_mask_full_sized(img_mask_pred_array[index][0], img.shape, output_folder=output_folder, image_name = image_name)
    if mask_folder:
        if plot:
            f, ax = plt.subplots(1, 3)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_true, cmap='Greys_r');  ax[1].axis("off"); 
            ax[2].imshow(mask_pred, cmap='Greys_r'); ax[2].axis("off"); plt.show()
        return img, mask_true, mask_pred
    else:
        if plot:
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_pred, cmap='Greys_r'); ax[1].axis("off"); plt.show()
        return img, mask_pred
