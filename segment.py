import os
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle as pkl
import ISIC_dataset as ISIC
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean
import models
np.random.seed(4)
K.set_image_dim_ordering('th')  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow

# Abstract: https://arxiv.org/abs/1703.04819

# Extract challenge Training / Validation / Test images as below
# Download from https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a
training_folder = "datasets/ISIC-2017_Training_Data"
training_mask_folder = "datasets/ISIC-2017_Training_Part1_GroundTruth"
training_labels_csv = "datasets/ISIC-2017_Training_Part3_GroundTruth.csv"
training_split_yml = "datasets/isic.yml"
validation_folder = "datasets/ISIC-2017_Validation_Data"    
test_folder = "datasets/ISIC-2017_Test_v2_Data"

# Place ISIC full dataset as below (optional)
isicfull_folder = "datasets/ISIC_Archive/image"
isicfull_mask_folder = "datasets/ISIC_Archive/mask"
isicfull_train_split="datasets/ISIC_Archive/train.txt"
isicfull_val_split="datasets/ISIC_Archive/val.txt"
isicfull_test_split="datasets/ISIC_Archive/test.txt"

# Folder to store predicted masks
validation_predicted_folder = "results/ISIC-2017_Validation_Predicted"
test_predicted_folder = "results/ISIC-2017_Test_v2_Predicted"

seed = 1
height, width = 128, 128
nb_epoch = 220
model_name = "model1"

do_train = True # train network and save as model_name
do_predict = True # use model to predict and save generated masks for Validation/Test
do_ensemble = False # use previously saved predicted masks from multiple models to generate final masks
ensemble_pkl_filenames = ["model1","model2", "model3", "model4"]
model = 'unet'
batch_size = 4
loss_param = 'dice'
optimizer_param = 'adam'
monitor_metric = 'val_jacc_coef'
fc_size = 8192
mean_type = 'imagenet' # 'sample' 'samplewise'
rescale_mask = True
use_hsv = False
dataset='isic' # 'isic' 'isicfull' 'isic_noval_notest' 'isic_other_split' 'isic_notest'
initial_epoch = 0 

metrics = [jacc_coef]
if use_hsv:
    n_channels = 6
    print "Using HSV"
else:
    n_channels = 3

print "Using {} mean".format(mean_type)
remove_mean_imagenet=False
remove_mean_samplewise=False
remove_mean_dataset=False
if mean_type == 'imagenet':
    remove_mean_imagenet = True;
elif mean_type == 'sample':
    remove_mean_samplewise = True
elif mean_type == 'dataset':
    remove_mean_dataset = True
    train_mean = np.array([[[ 180.71656799]],[[ 151.13494873]],[[ 139.89967346]]]);
    train_std = np.array([[[1]],[[1]],[[ 1]]]); # not using std
else:
    raise Exception("Wrong mean type")
    
loss_options = {'BCE': 'binary_crossentropy', 'dice':dice_loss, 'jacc':jacc_loss, 'mse':'mean_squared_error'}
optimizer_options = {'adam': Adam(lr=1e-5),
                     'sgd': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)}

loss = loss_options[loss_param]
optimizer = optimizer_options[optimizer_param]
model_filename = "weights/{}.h5".format(model_name)

print 'Create model'  

if model == 'unet':
    model = models.Unet(height,width, loss=loss, optimizer = optimizer, metrics = metrics, fc_size = fc_size, channels=n_channels)
elif model == 'unet2':
    model = models.Unet2(height,width, loss=loss, optimizer = optimizer, metrics = metrics, fc_size = fc_size, channels=n_channels)
elif model == 'vgg':
    model = models.VGG16(height,width, pretrained=True, freeze_pretrained = False, loss = loss, optimizer = optimizer, metrics = metrics)
else:
    print "Incorrect model name"

def myGenerator(train_generator, train_mask_generator, 
                remove_mean_imagenet=True, rescale_mask=True, use_hsv=False):
    while True:
        train_gen = next(train_generator)
        train_mask = next(train_mask_generator)
                
        if False: # use True to show images
            mask_true_show = np.where(train_mask>=0.5, 1, 0)
            mask_true_show = mask_true_show * 255
            mask_true_show = mask_true_show.astype(np.uint8)
            for i in range(train_gen.shape[0]):
                mask = train_mask[i].reshape((width,height))
                img=train_gen[i]
                img = img[0:3]
                img = img.astype(np.uint8)
                img = img.transpose(1,2,0)
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(img); ax[0].axis("off");
                ax[1].imshow(mask, cmap='Greys_r'); ax[1].axis("off"); plt.show()
        yield (train_gen, train_mask)

if do_train:
    if dataset == 'isicfull':
        n_samples = 2000 # per epoch
        print "Using ISIC full dataset"
        train_list, val_list, test_list = ISIC.train_val_test_from_txt(isicfull_train_split, isicfull_val_split, isicfull_test_split)
        # folders for resized images
        base_folder = "datasets/isicfull_{}_{}".format(height,width)
        image_folder = os.path.join(base_folder,"image")
        mask_folder = os.path.join(base_folder,"mask")
        if not os.path.exists(base_folder):
            print "Begin resizing..."
            ISIC.resize_images(train_list+val_list+test_list, input_image_folder=isicfull_folder, 
                          input_mask_folder=isicfull_mask_folder, 
                              output_image_folder=image_folder.format(height,width), output_mask_folder=mask_folder, 
                              height=height, width=width)
            print "Done resizing..."
            
    else:
        print "Using ISIC 2017 dataset"
        # folders for resized images
        base_folder = "datasets/isic_{}_{}".format(height, width)
        image_folder = os.path.join(base_folder,"image")
        mask_folder = os.path.join(base_folder,"mask")
        train_list, train_label, test_list, test_label = ISIC.train_test_from_yaml(yaml_file = training_split_yml, csv_file = training_labels_csv)
        train_list, val_list, train_label, val_label = ISIC.train_val_split(train_list, train_label, seed = seed, val_split = 0.20)
        if not os.path.exists(base_folder):
            ISIC.resize_images(train_list+val_list+test_list, 
                          input_image_folder=training_folder, input_mask_folder=training_mask_folder, 
                          output_image_folder=image_folder, output_mask_folder=mask_folder, 
                          height=height, width=width)
        if dataset == "isic_notest": # previous validation split will be used for training
            train_list = train_list + val_list
            val_list = test_list
        elif dataset=="isic_noval_notest": # previous validation/test splits will be used for training
            monitor_metric = 'jacc_coef'
            train_list = train_list + val_list + test_list 
            val_list = test_list
        elif dataset=="isic_other_split": # different split, uses previous val/test for training
            seed = 82
            train_list1, train_list2, train_label1, train_label2 = ISIC.train_val_split(train_list, train_label, seed=seed, val_split=0.30)
            train_list = val_list+test_list+train_list1 
            val_list = train_list2
            test_list = val_list        
        n_samples = len(train_list)
    
    print "Loading images"
    train, train_mask = ISIC.load_images(train_list, height, width, 
                                          image_folder, mask_folder, 
                                      remove_mean_imagenet=remove_mean_imagenet, 
                                   rescale_mask=rescale_mask, use_hsv=use_hsv, remove_mean_samplewise=remove_mean_samplewise)
    val, val_mask = ISIC.load_images(val_list, height, width, 
                                      image_folder, mask_folder,  
                                      remove_mean_imagenet=remove_mean_imagenet, 
                               rescale_mask=rescale_mask, use_hsv=use_hsv, remove_mean_samplewise=remove_mean_samplewise)
    test, test_mask = ISIC.load_images(test_list, height, width, 
                                      image_folder, mask_folder,
                                      remove_mean_imagenet=remove_mean_imagenet, 
                                 rescale_mask=rescale_mask, use_hsv=use_hsv, remove_mean_samplewise=remove_mean_samplewise)
    print "Done loading images"
    
    if remove_mean_dataset:
        print "\nUsing Train Mean: {} Std: {}".format(train_mean, train_std)
        train = (train-train_mean)/train_std
        val = (val-train_mean)/train_std
        test = (test-train_mean)/train_std

    print "Using batch size = {}".format(batch_size)
    print 'Fit model'
    model_checkpoint = ModelCheckpoint(model_filename, monitor=monitor_metric, save_best_only=True, verbose=1)
    data_gen_args = dict(featurewise_center=False, 
                            samplewise_center=remove_mean_samplewise, 
                            featurewise_std_normalization=False, 
                            samplewise_std_normalization=False, 
                            zca_whitening=False, 
                            rotation_range=270, 
                            width_shift_range=0.1, 
                            height_shift_range=0.1, 
                            horizontal_flip=False, 
                            vertical_flip=False, 
                            zoom_range=0.2,
                            channel_shift_range=0,
                            fill_mode='reflect',
                        dim_ordering=K.image_dim_ordering())
    data_gen_mask_args = dict(data_gen_args.items() + {'fill_mode':'nearest','samplewise_center':False}.items())
    print "Create Data Generator" 
    train_datagen = ImageDataGenerator(**data_gen_args)
    train_mask_datagen = ImageDataGenerator(**data_gen_mask_args)
    train_generator = train_datagen.flow(train, batch_size=batch_size, seed=seed)
    train_mask_generator = train_mask_datagen.flow(train_mask, batch_size=batch_size, seed=seed)
    train_generator_f = myGenerator(train_generator, train_mask_generator, 
                                   remove_mean_imagenet=remove_mean_imagenet,
                                    rescale_mask=rescale_mask, use_hsv=use_hsv)
    
    if dataset=="isic_noval_notest":
        print "Not using validation during training"
        history = model.fit_generator(
            train_generator_f,
            samples_per_epoch=n_samples,
           nb_epoch=nb_epoch, 
          callbacks=[model_checkpoint], initial_epoch=initial_epoch)
    else:
        history = model.fit_generator(
            train_generator_f,
            samples_per_epoch=n_samples,
            nb_epoch=nb_epoch, validation_data=(val,val_mask), 
            callbacks=[model_checkpoint], initial_epoch=initial_epoch)

    train = None; train_mask = None # clear memory
    print "Load best checkpoint"
    model.load_weights(model_filename) # load best saved checkpoint

    # evaluate model
    mask_pred_val = model.predict(val) 
    mask_pred_test = model.predict(test)
    for pixel_threshold in [0.5]: #np.arange(0.3,1,0.05):
        mask_pred_val = np.where(mask_pred_val>=pixel_threshold, 1, 0)
        mask_pred_val = mask_pred_val * 255
        mask_pred_val = mask_pred_val.astype(np.uint8)
        print "Validation Predictions Max: {}, Min: {}".format(np.max(mask_pred_val), np.min(mask_pred_val))
        print model.evaluate(val, val_mask, batch_size = batch_size, verbose=1) 
        dice, jacc = dice_jacc_mean(val_mask, mask_pred_val, smooth = 0)
        print model_filename
        print "Resized val dice coef      : {:.4f}".format(dice)
        print "Resized val jacc coef      : {:.4f}".format(jacc)

        mask_pred_test = np.where(mask_pred_test>=pixel_threshold, 1, 0)
        mask_pred_test = mask_pred_test * 255
        mask_pred_test = mask_pred_test.astype(np.uint8)
        print model.evaluate(test, test_mask, batch_size = batch_size, verbose=1) 
        dice, jacc = dice_jacc_mean(test_mask, mask_pred_test, smooth = 0)
        print "Resized test dice coef      : {:.4f}".format(dice)
        print "Resized test jacc coef      : {:.4f}".format(jacc)
else:
    print 'Load model'
    model.load_weights(model_filename)

def predict_challenge(challenge_folder, challenge_predicted_folder, mask_pred_challenge=None, plot=True):
    challenge_list = ISIC.list_from_folder(challenge_folder)
    challenge_resized_folder = challenge_folder+"_{}_{}".format(height,width)
    
    if not os.path.exists(challenge_resized_folder):
        ISIC.resize_images(challenge_list, input_image_folder=challenge_folder, input_mask_folder=None, 
                          output_image_folder=challenge_resized_folder, output_mask_folder=None, 
                          height=height, width=width)

    challenge_resized_list =  [name.split(".")[0]+".png" for name in challenge_list]
    challenge_images = ISIC.load_images(challenge_resized_list, 
            height, width, image_folder=challenge_resized_folder,
            mask_folder=None, remove_mean_imagenet=True, use_hsv = use_hsv,remove_mean_samplewise=remove_mean_samplewise)
    
    if remove_mean_dataset:
        challenge_images = (challenge_images-train_mean)/train_std
    if mask_pred_challenge is None:
        mask_pred_challenge = model.predict(challenge_images)
    with open('{}.pkl'.format(os.path.join(challenge_predicted_folder,model_name)), 'wb') as f:
        pkl.dump(mask_pred_challenge, f)
    mask_pred_challenge = np.where(mask_pred_challenge>=0.5, 1, 0)
    mask_pred_challenge = mask_pred_challenge * 255
    mask_pred_challenge = mask_pred_challenge.astype(np.uint8)

    challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name)
    if not os.path.exists(challenge_predicted_folder):
        os.makedirs(challenge_predicted_folder)

    print "Start challenge prediction:"

    for i in range(len(challenge_list)):
        print "{}: {}".format(i, challenge_list[i])
        ISIC.show_images_full_sized(image_list = challenge_list, img_mask_pred_array = mask_pred_challenge, 
                image_folder=challenge_folder, mask_folder=None, index = i, output_folder=challenge_predicted_folder, plot=plot)

def join_predictions(pkl_folder, pkl_files, binary=False, threshold=0.5):
    n_pkl = float(len(pkl_files))
    array = None
    for fname in pkl_files:
        with open(os.path.join(pkl_folder,fname+".pkl"), "rb") as f:
            tmp = pkl.load(f)
            if binary:
                tmp = np.where(tmp>=threshold, 1, 0)
            if array is None:
                array = tmp
            else:
                array = array + tmp
    return array/n_pkl
    
if do_predict:
    # free memory
    train = None
    train_mask = None
    val = None
    test = None 
    
    print "Start Challenge Validation"    
    predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder, plot=False)
    print "Start Challenge Test"    
    predict_challenge(challenge_folder=test_folder, challenge_predicted_folder=test_predicted_folder, plot=False)
    
if do_ensemble:
    threshold = 0.5
    binary = False
    val_array = join_predictions(pkl_folder = validation_predicted_folder, pkl_files=ensemble_pkl_filenames, binary=binary, threshold=threshold)
    test_array = join_predictions(pkl_folder = test_predicted_folder, pkl_files=ensemble_pkl_filenames, binary=binary, threshold=threshold)
    model_name="ensemble_{}".format(threshold)
    for f in ensemble_pkl_filenames:
        model_name = model_name + "_" + f
    print "Predict Validation:"
    plot = True
    predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder,
                        mask_pred_challenge=val_array, plot=plot)
    print "Predict Test:"
    plot = False
    predict_challenge(challenge_folder=test_folder, challenge_predicted_folder=test_predicted_folder,
                    mask_pred_challenge=test_array, plot=plot)
