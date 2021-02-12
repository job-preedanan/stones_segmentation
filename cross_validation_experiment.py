from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard, Callback
import config_file as cfg
import segmentation_loss_function as metrics
import numpy as np
import cv2
import os
import pandas as pd
from myUnet import UNet
from stone_augment import stone_embedding
from segmentation_evaluate import segmentation_evaluate, segmentation_evaluate_full, metrics_compute
from KUB_partitions_function import create_KUB_partitions, combine_KUB_partitions


IMAGE_DIR = 'data/all_images_full'
MASK_DIR = 'data/all_groundtruth'
KUB_MAP_DIR = 'data/Full_KUB_map'

K_STONE_DIR = 'data/cropped_stone_dataset_new/k_stone_dataset'
K_STONE_DIST_DIR = 'data/cropped_stone_dataset_new/k_stone_dist_map'
U_STONE_DIR = 'data/cropped_stone_dataset_new/u_stone_dataset'
U_STONE_DIST_DIR = 'data/cropped_stone_dataset_new/u_stone_dist_map'
B_STONE_DIR = 'data/cropped_stone_dataset_new/b_stone_dataset'
B_STONE_DIST_DIR = 'data/cropped_stone_dataset_new/b_stone_dist_map'

IMAGE_SIZE = cfg.image_params['image_size']


# Normalize Image
def normalize_x(image):
    image = image / 127.5 - 1
    return image


# Normalize GT
def normalize_y(image):
    image = image / 255
    return image


def denormalize_x(image):
    image = (image + 1) * 127.5
    return image


# Denormarlize for saving results
def denormalize_y(image):
    image = image * 255
    return image


def load_images(images_list, image_type, input_size, partition=True):

    # load partition images
    if partition:
        images = np.zeros((len(images_list) * 3, input_size, input_size, 1), np.float32)
        for i, image_name in enumerate(images_list):
            if image_type == 0:  # image
                image = cv2.imread(os.path.join(IMAGE_DIR, image_name[0]), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (round(image.shape[1] / (image.shape[0] / 1024)), 1024))

                # read KUB map
                KUB_map = cv2.imread(os.path.join(KUB_MAP_DIR, image_name[0][:-4] + '.png'))
                KUB_map = cv2.resize(KUB_map, (image.shape[1], image.shape[0]))

                # create 3 partitions
                L_partition, R_partition, B_partition = create_KUB_partitions(image, KUB_map)
                # L_partition, R_partition, B_partition = create_KUB_partitions_old(image)

                L_partition = cv2.resize(normalize_x(L_partition), (input_size, input_size))
                images[3*i] = L_partition[:, :, np.newaxis]

                R_partition = cv2.resize(normalize_x(R_partition), (input_size, input_size))
                images[3*i + 1] = R_partition[:, :, np.newaxis]

                B_partition = cv2.resize(normalize_x(B_partition), (input_size, input_size))
                images[3*i + 2] = B_partition[:, :, np.newaxis]

                # if s < 20:
                #     save_path = cfg.path['save_dir']['root_dir'] + os.sep + cfg.path['save_dir'][
                #         'save_folder'] + os.sep + cross_val_folder
                #     cv2.imwrite(save_path + '/partition/' + image_name[0][:-4] + '_L.png', denormalize_x(L_partition))
                #     cv2.imwrite(save_path + '/partition/' + image_name[0][:-4] + '_R.png', denormalize_x(R_partition))
                #     cv2.imwrite(save_path + '/partition/' + image_name[0][:-4] + '_B.png', denormalize_x(B_partition))

            elif image_type == 1:  # mask
                if image_name[2] == 1:    # sc
                    image = cv2.imread(os.path.join(MASK_DIR, image_name[1]), cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (round(image.shape[1] / (image.shape[0] / 1024)), 1024))

                    # read KUB map
                    KUB_map = cv2.imread(os.path.join(KUB_MAP_DIR, image_name[0][:-4] + '.png'))
                    KUB_map = cv2.resize(KUB_map, (image.shape[1], image.shape[0]))

                    # create 3 partitions
                    L_partition, R_partition, B_partition = create_KUB_partitions(image, KUB_map)
                    # L_partition, R_partition, B_partition = create_KUB_partitions_old(image)

                    L_partition = cv2.resize(normalize_y(L_partition), (input_size, input_size))
                    images[3 * i] = L_partition[:, :, np.newaxis]

                    R_partition = cv2.resize(normalize_y(R_partition), (input_size, input_size))
                    images[3 * i + 1] = R_partition[:, :, np.newaxis]

                    B_partition = cv2.resize(normalize_y(B_partition), (input_size, input_size))
                    images[3 * i + 2] = B_partition[:, :, np.newaxis]

                    save_path = cfg.path['save_dir']['root_dir'] + os.sep + cfg.path['save_dir'][
                        'save_folder'] + os.sep + cross_val_folder
                    # if s < 20:
                    #     cv2.imwrite(save_path + '/partition_gt/' + image_name[0][:-4] + '_L.png', denormalize_y(L_partition))
                    #     cv2.imwrite(save_path + '/partition_gt/' + image_name[0][:-4] + '_R.png', denormalize_y(R_partition))
                    #     cv2.imwrite(save_path + '/partition_gt/' + image_name[0][:-4] + '_B.png', denormalize_y(B_partition))

    # load full images
    else:
        images = np.zeros((len(images_list), input_size, input_size, 1), np.float32)
        for i, image_name in enumerate(images_list):
            if image_type == 0:  # image
                image = cv2.imread(os.path.join(IMAGE_DIR, image_name[0]), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (input_size, input_size))
                image = image[:, :, np.newaxis]
                images[i] = normalize_x(image)

            elif image_type == 1:  # mask
                if image_name[2] == 1:    # sc
                    image = cv2.imread(os.path.join(MASK_DIR, image_name[1]), cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (input_size, input_size))
                    image = image[:, :, np.newaxis]
                    images[i] = normalize_y(image)

    return images


def stone_augmenting_generator(image, mask, stone_location_map, stone_num, prob_map=False):
    import random
    import imutils

    # stone list from cropped stone databased
    k_stone_list = os.listdir(K_STONE_DIR)
    u_stone_list = os.listdir(U_STONE_DIR)
    b_stone_list = os.listdir(B_STONE_DIR)

    # get all location map coordinate list
    # select each partitions separately
    location_k = random.choice(np.argwhere(stone_location_map[:, :, 2] == 255))
    location_u = random.choice(np.argwhere(stone_location_map[:, :, 1] == 255))
    location_b = random.choice(np.argwhere(stone_location_map[:, :, 0] == 255))

    if prob_map:
        location_list = np.argwhere(stone_location_map > 0)       # probability map - indice
        prob_map = stone_location_map[stone_location_map > 0]     # probability map - values

    new_image = image
    new_mask = mask
    for i in range(stone_num):

        # random stone coordinate(x,y) in location list : uniform distribution
        location = random.choice(np.argwhere(stone_location_map == 255))

        if prob_map:
            location = random.choices(location_list, weights=prob_map)
            location = location[0]

        r_b = 0
        # random stone from database  - load stone and its distance map
        if location[2] == 2:              # kidney stone
            r_b = 1
            stone_name = random.choice(k_stone_list)
            stone = cv2.imread((K_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            stone_dist_map = cv2.imread((K_STONE_DIST_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

        elif location[2] == 1:              # ureter stone
            r_b = 1
            stone_name = random.choice(u_stone_list)
            stone = cv2.imread((U_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            stone_dist_map = cv2.imread((U_STONE_DIST_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

        elif location[2] == 0:              # bladder region
            r_b = 1
            stone_name = random.choice(b_stone_list)
            stone = cv2.imread((B_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            stone_dist_map = cv2.imread((B_STONE_DIST_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

        # resize cropped stone
        stone = cv2.resize(stone, (round(stone.shape[0] / 1), round(stone.shape[1] / 1)))
        stone_dist_map = cv2.resize(stone_dist_map, stone.shape[:2])

        # stone rotating
        if r_b:
            angle = random.randrange(-5, 5)
            stone = imutils.rotate(stone, angle)  # image
            stone_dist_map = imutils.rotate(stone_dist_map, angle)  # image

        # horizontal flip
        if bool(random.getrandbits(1)) and r_b:
            stone = cv2.flip(stone, 1)
            stone_dist_map = cv2.flip(stone_dist_map, 1)

        # vertical flip
        if bool(random.getrandbits(1)) and r_b:
            stone = cv2.flip(stone, 0)
            stone_dist_map = cv2.flip(stone_dist_map, 0)

        # random stone coordinate in location list
        if r_b:
            new_image, new_mask = stone_embedding(image, mask, stone, stone_dist_map, location)

    return new_image, new_mask


def image_mask_augmentation(image, mask, stone_location_map, datagen_methods, img_name, image_type, epoch, cross_val_folder):
    import random
    import imutils

    augmented_image = image
    augmented_mask = mask

    save = False

    # stone embedding
    stone_augment = datagen_methods['stone_augment']
    if stone_augment:
        if image_type == 0:    # SF
            save = True
            stone_num = random.randint(1, 3)  # random stone number
            if stone_num != 0:
                # save = True
                augmented_image, augmented_mask = stone_augmenting_generator(augmented_image,
                                                                             augmented_mask,
                                                                             stone_location_map,
                                                                             stone_num)

    # image rotation
    angle_range = datagen_methods['rotation_range']
    if angle_range != False:
        angle = random.randrange(-angle_range, angle_range)
        augmented_image = imutils.rotate(image, angle)  # image
        augmented_mask = imutils.rotate(mask, angle)  # mask
        stone_location_map = imutils.rotate(stone_location_map, angle)  # KUB map

    # image horizontal flip
    horizontal_flip = datagen_methods['horizontal_flip']
    if horizontal_flip != False:
        if bool(random.getrandbits(1)):  # random flip/not flip
            augmented_image = cv2.flip(augmented_image, 1)
            augmented_mask = cv2.flip(augmented_mask, 1)
            stone_location_map = cv2.flip(stone_location_map, 1)  # KUB map

    # image vertical flip
    vertical_flip = datagen_methods['vertical_flip']
    if vertical_flip != False:
        if bool(random.getrandbits(1)):  # random flip/not flip
            augmented_image = cv2.flip(augmented_image, 0)  # image
            augmented_mask = cv2.flip(augmented_mask, 0)  # mask

    # image zooming in (=> crop and resize back)
    def image_zooming(img, zoom_value):
        h, w = img.shape[:2]
        # size of cropped img
        crop_h = h * zoom_value
        crop_w = w * zoom_value
        # border size to cut
        cut_h = round((h - crop_h) / 2)
        cut_w = round((w - crop_w) / 2)
        # crop
        cropped = img[cut_h:h - cut_h, cut_w:w - cut_w]
        # resize back
        zoom_img = cv2.resize(cropped, (h, w))

        return zoom_img

    zoom = datagen_methods['zoom_range']  # percentage value
    if zoom != False:
        zoom_value = random.randrange(100 - zoom, 100) / 100  # 0.xx -1.00 times
        augmented_image = image_zooming(augmented_image, zoom_value)
        augmented_mask = image_zooming(augmented_mask, zoom_value)
        stone_location_map = image_zooming(stone_location_map, zoom_value)  # KUB map

    # intensty shift ** implement only image
    def intensity_shift_funct(img, brightness_range):
        value = random.randrange(-brightness_range, brightness_range) / 255
        img = img + value
        return np.clip(img, 0, 255)

    brightness_shift = datagen_methods['brightness_shift']  # uint8 value
    if brightness_shift != False:
        augmented_image = intensity_shift_funct(augmented_image, brightness_range=brightness_shift)

    # image contrast change  ** implement only image
    def contrast_adjust(img, contrast_range):
        alpha = random.randrange(100 - contrast_range, 100 + contrast_range) / 100
        return img * alpha

    contrast_change = datagen_methods['contrast_change']  # percentage value
    if contrast_change != False:
        augmented_image = contrast_adjust(augmented_image, contrast_range=contrast_change)

    # resize image,gt
    # augmented_image = cv2.resize(augmented_image, (IMAGE_SIZE, IMAGE_SIZE))
    # augmented_mask = cv2.resize(augmented_mask, (IMAGE_SIZE, IMAGE_SIZE))

    # save only fist 5 epochs
    if epoch <= 1 and save:
        save_path = cfg.path['save_dir']['root_dir'] + os.sep + cfg.path['save_dir']['save_folder'] + os.sep + cross_val_folder
        cv2.imwrite(save_path + '/augments/' + img_name[:-4] + '_' + str(epoch) + '_s' + str(stone_num) + '.png', augmented_image)
        cv2.imwrite(save_path + '/augments_gt/' + img_name[:-4] + '_' + str(epoch) + '_s' + str(stone_num) + '.png', augmented_mask)

    return augmented_image, augmented_mask, stone_location_map


def generator(sc_samples, sf_samples, batch_size, datagen_methods, shuffle_data, image_size, cross_val_folder, partition):
    import random

    epoch = 0
    while True:  # loop generator forever

        # list moving function
        def list_index_move(list, split_num):
            split_idx = int(round(len(list) * split_num))  # split index
            new_list = list[split_idx:]
            new_list = new_list + list[:split_idx]
            select_list = list[:split_idx]
            return new_list, select_list

        sf_samples, select_sf = list_index_move(sf_samples, split_num=0.25)

        # combine sc + selected sf
        samples = sc_samples + select_sf

        # shuffle images list
        if shuffle_data:
            # random sample[images,gt] lists
            random.shuffle(samples)

        epoch += 1
        num_samples = len(samples)
        # for each batch
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples using in this batch
            batch_samples = samples[offset:offset + batch_size]

            # Initialise X_train and y_train arrays for this batch
            if partition:
                x_train = np.zeros((batch_size*3, image_size, image_size, 1), np.float32)
                y_train = np.zeros((batch_size*3, image_size, image_size, 1), np.float32)
            else:
                x_train = np.zeros((batch_size, image_size, image_size, 1), np.float32)
                y_train = np.zeros((batch_size, image_size, image_size, 1), np.float32)

            # For each sample
            for i, batch_sample in enumerate(batch_samples):
                # load image (X) and gt mask (y)
                img_name = batch_sample[0]
                gt_name = batch_sample[1]
                img_type = batch_sample[2]

                # read full images
                img = cv2.imread(os.path.join(IMAGE_DIR, img_name), cv2.IMREAD_GRAYSCALE)
                if img_type == 0:      #sf
                    gt = np.zeros(img.shape)

                elif img_type == 1:    #sc
                    gt = cv2.imread(os.path.join(MASK_DIR, gt_name), cv2.IMREAD_GRAYSCALE)

                # resize image,gt for having same size -> prepare for augmentation
                img = cv2.resize(img, (round(img.shape[1]/(img.shape[0]/1024)), 1024))
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]))

                # read KUB map
                KUB_map = cv2.imread(os.path.join(KUB_MAP_DIR, img_name[:-4] + '.png'))
                KUB_map = cv2.resize(KUB_map, (img.shape[1], img.shape[0]))

                # ** image augmentation : augment image and gt together
                img, gt, KUB_map = image_mask_augmentation(img,
                                                           gt,
                                                           KUB_map,
                                                           datagen_methods,
                                                           img_name,
                                                           img_type,
                                                           epoch,
                                                           cross_val_folder)

                # partitions
                if partition:
                    # create 3 partitions (org + gt)
                    L_img, R_img, B_img = create_KUB_partitions(img, KUB_map)
                    L_gt, R_gt, B_gt = create_KUB_partitions(gt, KUB_map)
                    # L_img, R_img, B_img = create_KUB_partitions_old(img)
                    # L_gt, R_gt, B_gt = create_KUB_partitions_old(gt)

                    # L partition
                    L_img = cv2.resize(normalize_x(L_img), (IMAGE_SIZE, IMAGE_SIZE))  # [-1,1]
                    x_train[3*i] = L_img[:, :, np.newaxis]
                    L_gt = cv2.resize(normalize_y(L_gt), (IMAGE_SIZE, IMAGE_SIZE))  # [0,1]
                    y_train[3*i] = L_gt[:, :, np.newaxis]

                    # R partition
                    R_img = cv2.resize(normalize_x(R_img), (IMAGE_SIZE, IMAGE_SIZE))  # [-1,1]
                    x_train[3*i + 1] = R_img[:, :, np.newaxis]
                    R_gt = cv2.resize(normalize_y(R_gt), (IMAGE_SIZE, IMAGE_SIZE))    # [0,1]
                    y_train[3*i + 1] = R_gt[:, :, np.newaxis]

                    # B partition
                    B_img = cv2.resize(normalize_x(B_img), (IMAGE_SIZE, IMAGE_SIZE))  # [-1,1]
                    x_train[3*i + 2] = B_img[:, :, np.newaxis]
                    B_gt = cv2.resize(normalize_y(B_gt), (IMAGE_SIZE, IMAGE_SIZE))  # [0,1]
                    y_train[3*i + 2] = B_gt[:, :, np.newaxis]

                    save_path = cfg.path['save_dir']['root_dir'] + os.sep + cfg.path['save_dir'][
                        'save_folder'] + os.sep + cross_val_folder

                    if epoch == 1 and offset < 24:
                        cv2.imwrite(save_path + '/partition/' + img_name[:-4] + '_L.png', denormalize_x(L_img))
                        cv2.imwrite(save_path + '/partition_gt/' + img_name[:-4] + '_L.png', denormalize_y(L_gt))
                        cv2.imwrite(save_path + '/partition/' + img_name[:-4] + '_R.png', denormalize_x(R_img))
                        cv2.imwrite(save_path + '/partition_gt/' + img_name[:-4] + '_R.png', denormalize_y(R_gt))
                        cv2.imwrite(save_path + '/partition/' + img_name[:-4] + '_B.png', denormalize_x(B_img))
                        cv2.imwrite(save_path + '/partition_gt/' + img_name[:-4] + '_B.png', denormalize_y(B_gt))

                # full image
                else:
                    # normalize img+gt, add 1 axis, and add example to arrays
                    img = normalize_x(img)  # [-1,1]
                    img = img[:, :, np.newaxis]
                    x_train[i] = img
                    gt = normalize_y(gt)  # [0,1]
                    gt = gt[:, :, np.newaxis]
                    y_train[i] = gt

            # The generator-y part: yield the next training batch
            yield x_train, y_train


def learning_rate_decay(lr_decay_params, INITIAL_LR, LIMIT_LR):
    # (1) learning rate decay function
    def decay_funct(epoch):
        new_lr = INITIAL_LR / (1 + lr_decay_params['decay_fn']['decay_val'] * epoch)
        if new_lr <= LIMIT_LR:
            new_lr = LIMIT_LR

        return new_lr

    lr_function = LearningRateScheduler(decay_funct, 1)

    # (2) learning rate reduce(half) by condition
    lr_reduce = ReduceLROnPlateau(monitor=lr_decay_params['conditional_fn']['monitor'],
                                  factor=lr_decay_params['conditional_fn']['factor'],
                                  patience=10,
                                  min_lr=LIMIT_LR)

    lr_decay_method = lr_reduce  # default
    if cfg.params['lr_decay']['method'] == 0:
        lr_decay_method = lr_reduce
    elif cfg.params['lr_decay']['method'] == 1:
        lr_decay_method = lr_function

    return lr_decay_method


def plot_summary_graph(history, save_path):
    import matplotlib.pyplot as plt

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_loss.png')
    plt.clf()

    # summarize history for accuracy
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice coefficient')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_acc.png')
    plt.clf()

    # summarize history for recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_recall.png')
    plt.clf()

    # summarize history for precision
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path + '/model_precision.png')
    plt.clf()


def train(sc_train_samples, sf_train_samples, test_samples, cross_val_folder):

    # define parameters from configuration file
    BATCH_SIZE = cfg.params['batch_size']

    # define data augmentation methods
    datagen = dict(stone_augment=True,
                   rotation_range=5,
                   zoom_range=False,
                   vertical_flip=False,
                   horizontal_flip=True,
                   brightness_shift=False,
                   contrast_change=False)

    # create a generator object for training samples
    train_generator = generator(sc_train_samples,
                                sf_train_samples,
                                batch_size=BATCH_SIZE,
                                datagen_methods=datagen,
                                shuffle_data=True,
                                image_size=IMAGE_SIZE,
                                cross_val_folder=cross_val_folder,
                                partition=True)

    # load validating samples  // image_type: 0-image, 1-gt
    x_val = load_images(test_samples, image_type=0, input_size=IMAGE_SIZE, partition=True)
    y_val = load_images(test_samples, image_type=1, input_size=IMAGE_SIZE, partition=True)

    print('#train = ' + str(3*len(sc_train_samples)))
    print('#validate = ' + str(len(x_val)))

    # load U-Net model
    print('Load model ...')
    input_channel_count = cfg.model_params['input_channel']
    output_channel_count = cfg.model_params['output_channel']
    first_layer_filter_count = cfg.model_params['first_layer_filters']

    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    # model.summary()

    # define loss, optimizer, and monitoring metrics to model
    model.compile(loss=cfg.params['loss_function'],
                  optimizer=Adam(lr=cfg.params['initial_lr']),
                  metrics=[metrics.dice_coef, metrics.recall, metrics.precision])

    # define early stopping
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       min_delta=0.001,
                       patience=50,
                       restore_best_weights=True)

    # define learning rate decay function
    lr_decay_method = learning_rate_decay(cfg.params['lr_decay'], cfg.params['initial_lr'], cfg.params['limit_lr'])

    # fits the model on batches with real-time data augmentation:
    total_samples = len(sc_train_samples) + round(len(sf_train_samples)/4)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=total_samples / BATCH_SIZE,
                                  epochs=cfg.params['epoch'],
                                  callbacks=[es, lr_decay_method],
                                  validation_data=(x_val, y_val))

    save_path = cfg.path['save_dir']['root_dir'] + os.sep + cfg.path['save_dir']['save_folder'] + os.sep + cross_val_folder
    model.save_weights(save_path + '/unet_weights.hdf5')

    # save graphs
    plot_summary_graph(history, save_path)

    # save all values from training process to xlsx
    loss_hist = pd.DataFrame(history.history)
    loss_hist.to_excel(save_path + '/history.xlsx')


# Predict Result
def predict(test_samples, cross_val_folder):
    import cv2

    # testData
    X_test = load_images(test_samples, image_type=0, input_size=IMAGE_SIZE, partition=True)
    Y_true = load_images(test_samples, image_type=1, input_size=IMAGE_SIZE, partition=False)

    # define U-Net parameters
    input_channel_count = cfg.model_params['input_channel']
    output_channel_count = cfg.model_params['output_channel']
    first_layer_filter_count = cfg.model_params['first_layer_filters']

    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()

    save_path = cfg.path['save_dir']['root_dir'] + '/' + cfg.path['save_dir']['save_folder'] + '/' + cross_val_folder
    model.load_weights(save_path + '/unet_weights.hdf5')
    BATCH_SIZE = 16

    # predict
    y_pred = model.predict(X_test, BATCH_SIZE)

    # combine y_pred
    partitions = True
    if partitions:
        n = 0
        y_pred = y_pred[:, :, :, 0]
        full_Y_pred = np.zeros((round(len(y_pred)/3), 1024, 1024), np.float32)
        for offset in range(0, len(y_pred), 3):
            partitions = y_pred[offset:offset + 3]
            full_img = cv2.imread(IMAGE_DIR + os.sep + test_samples[n][0])
            full_KUB_map = cv2.resize(cv2.imread(KUB_MAP_DIR + os.sep + test_samples[n][0][:- 4] + '.png'), (full_img.shape[1], full_img.shape[0]))
            full_Y_pred_combine = combine_KUB_partitions(partitions[0], partitions[1], partitions[2], full_KUB_map)
            # full_Y_pred_combine = combine_KUB_partitions_old(partitions[0], partitions[1], partitions[2])
            full_Y_pred[n] = cv2.resize(full_Y_pred_combine, (1024, 1024))
            n += 1
    else:
        full_Y_pred = y_pred[:, :, :, 0]

    evaluation_results_pixel = {'image_name': [],
                                'total_stones': [],
                                'TP': [],
                                'FP': [],
                                'FN': [],
                                'recall': [],
                                'precision': [],
                                'F1': [],
                                'F2': []}

    evaluation_results = {'image_name': [],
                          'total_stones': [],
                          'TP': [],
                          'FP': [],
                          'FN': [],
                          'recall': [],
                          'precision': [],
                          'F1': [],
                          'F2': []}

    stone_results = {'stone_name': [],
                     '(x,y)': [],
                     '(w,h)': [],
                     'stone_size': [],
                     'detect': []}

    sc_num = 0     # for counting sc images
    for i, full_y_pred in enumerate(full_Y_pred):

        image_name = test_samples[i][0][:- 4]

        # read org image
        img = cv2.imread(IMAGE_DIR + os.sep + test_samples[i][0])
        img = cv2.resize(img, (1024, 1024))

        # predicted image
        full_y_pred = cv2.resize(full_y_pred, (img.shape[1], img.shape[0]))
        heatmap = np.maximum(full_y_pred, 0)
        full_y_pred = denormalize_y(full_y_pred)
        full_y_pred = np.array(full_y_pred, dtype=np.uint8)

        # gt image
        y_true = cv2.resize(Y_true[i], (img.shape[1], img.shape[0]))
        y_true = denormalize_y(y_true)
        y_true = np.array(y_true, dtype=np.uint8)

        # evaluation function
        #full_KUB_map = cv2.resize(cv2.imread(KUB_MAP_DIR + os.sep + image_name + '.png'), (img.shape[1], img.shape[0]))
        #evaluated_data, stone_data = segmentation_evaluate(full_y_pred, y_true, full_KUB_map)
        evaluate_pixelbased, evaluate_regionbased, stone_data = segmentation_evaluate_full(full_y_pred, y_true)

        evaluation_results_pixel['image_name'].append(image_name)
        evaluation_results_pixel['total_stones'].append(evaluate_pixelbased[0])
        evaluation_results_pixel['TP'].append(evaluate_pixelbased[1])
        evaluation_results_pixel['FN'].append(evaluate_pixelbased[2])
        evaluation_results_pixel['FP'].append(evaluate_pixelbased[3])
        evaluation_results_pixel['recall'].append(metrics_compute(evaluate_pixelbased[1],
                                                                  evaluate_pixelbased[2],
                                                                  evaluate_pixelbased[3])[0])
        evaluation_results_pixel['precision'].append(metrics_compute(evaluate_pixelbased[1],
                                                                     evaluate_pixelbased[2],
                                                                     evaluate_pixelbased[3])[1])
        evaluation_results_pixel['F1'].append(metrics_compute(evaluate_pixelbased[1],
                                                              evaluate_pixelbased[2],
                                                              evaluate_pixelbased[3])[2])
        evaluation_results_pixel['F2'].append(metrics_compute(evaluate_pixelbased[1],
                                                              evaluate_pixelbased[2],
                                                              evaluate_pixelbased[3])[3])

        # STONE EVALUATION
        evaluation_results['image_name'].append(image_name)
        evaluation_results['total_stones'].append(evaluate_regionbased[0])
        evaluation_results['TP'].append(evaluate_regionbased[1])
        evaluation_results['FN'].append(evaluate_regionbased[2])
        evaluation_results['FP'].append(evaluate_regionbased[3])
        evaluation_results['recall'].append(metrics_compute(evaluate_regionbased[1],
                                                            evaluate_regionbased[2],
                                                            evaluate_regionbased[3])[0])
        evaluation_results['precision'].append(metrics_compute(evaluate_regionbased[1],
                                                               evaluate_regionbased[2],
                                                               evaluate_regionbased[3])[1])
        evaluation_results['F1'].append(metrics_compute(evaluate_regionbased[1],
                                                        evaluate_regionbased[2],
                                                        evaluate_regionbased[3])[2])
        evaluation_results['F2'].append(metrics_compute(evaluate_regionbased[1],
                                                        evaluate_regionbased[2],
                                                        evaluate_regionbased[3])[3])

        # STONE DATA
        for i in range(len(stone_data)):
            stone_results['stone_name'].append(image_name + '_' + stone_data[i][0])
            stone_results['(x,y)'].append((stone_data[i][1], stone_data[i][2]))
            stone_results['(w,h)'].append((stone_data[i][3], stone_data[i][4]))
            stone_results['stone_size'].append(stone_data[i][5])
            stone_results['detect'].append(stone_data[i][6])

        # heatmap generate
        heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        hif = .8
        y_pred_superimposed = heatmap * hif + img

        # write predicted image
        if test_samples[i][2] == 1:      #sc
            sc_num = sc_num + 1
            cv2.imwrite(save_path + '/PredictedResults/sc/' + image_name + '_Result.png', full_y_pred)
            cv2.imwrite(save_path + '/PredictedResults/sc_heatmap/' + image_name + '_Result.png', y_pred_superimposed)

        elif test_samples[i][2] == 0:    #sf
            cv2.imwrite(save_path + '/PredictedResults/sf/' + image_name + '_Result.png', full_y_pred)
            cv2.imwrite(save_path + '/PredictedResults/sf_heatmap/' + image_name + '_Result.png', y_pred_superimposed)

    data_pixel = pd.DataFrame(evaluation_results_pixel, columns=['image_name',
                                                                 'total_stones',
                                                                 'TP', 'FN', 'FP',
                                                                 'recall', 'precision', 'F1', 'F2'])
    data_pixel.to_excel(save_path + '/evaluation_pixelbased.xlsx', index=None, header=True)

    data1 = pd.DataFrame(evaluation_results, columns=['image_name',
                                                      'total_stones',
                                                      'TP', 'FN', 'FP',
                                                      'recall', 'precision', 'F1', 'F2'])

    data1.to_excel(save_path + '/evaluation_regionbased.xlsx', index=None, header=True)

    data2 = pd.DataFrame(stone_results, columns=['stone_name',
                                                 '(x,y)', '(w,h)',
                                                 'stone_size', 'detect'])
    data2.to_excel(save_path + '/stone_results.xlsx', index=None, header=True)


if __name__ == '__main__':
    import random

    # path
    folder_path = cfg.path['save_dir']['root_dir'] + os.sep + cfg.path['save_dir']['save_folder']

    # read excel file
    # excel = pd.read_excel('data/image_list_full.xlsx')
    # images_list = pd.DataFrame(excel, columns=['image', 'gt', 'stone']).values.tolist()
    # sc_list = images_list[0:1156]       # 1156 images (stone-contained)
    # sf_list = images_list[1156:2356]    # 1200 images  (stone-free)

    # random both lists and save to excel

    # random.shuffle(sc_list)
    # random.shuffle(sf_list)
    # sc_data = pd.DataFrame(sc_list)
    # sc_data.to_excel(folder_path + '/sc_data.xlsx', index=None, header=None)
    # sf_data = pd.DataFrame(sf_list)
    # sf_data.to_excel(folder_path + '/sf_data.xlsx', index=None, header=None)

    sc_list = pd.DataFrame(pd.read_excel(folder_path + '/sc_data.xlsx'), columns=['image', 'gt', 'stone']).values.tolist()
    sf_list = pd.DataFrame(pd.read_excel(folder_path + '/sf_data.xlsx'), columns=['image', 'gt', 'stone']).values.tolist()


    # k-fold cross validation
    num_cross_val = 5
    for i in range(num_cross_val):
        print('cross validation #' + str(i+1))
        cross_val_folder = 'cross_val#' + str(i+1)
        try:
            os.mkdir(folder_path + os.sep + cross_val_folder)
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'augments')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'augments_gt')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'partition')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'partition_gt')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sc')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sc_heatmap')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sf')
            os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sf_heatmap')
        except FileExistsError:
            print('folder exist')

        # data spliting function
        def split_train_test(samples, VAL_SPLIT):
            split_idx = int(round(len(samples) * VAL_SPLIT))  # split index
            test = samples[:split_idx]
            train = samples[split_idx:]
            return train, test

        # train/test split
        sc_train_samples, sc_test_samples = split_train_test(sc_list, VAL_SPLIT=cfg.params['val_split'])  #sc
        sf_train_samples, sf_test_samples = split_train_test(sf_list, VAL_SPLIT=cfg.params['val_split'])  #sf

        # combine sf with sc dataset (train and val)
        train_samples = sc_train_samples + sf_train_samples
        test_samples = sc_test_samples + sf_test_samples

        print('#train = ' + str(len(train_samples)))
        print('#test = ' + str(len(test_samples)))

        if i < 4:
            # training U-net  (validation set in training is only sc dataset)
            print('########## Training ###########')
            train(sc_train_samples, sf_train_samples, sc_test_samples, cross_val_folder)

            # testing
            print('########## Testing ###########')
            predict(test_samples, cross_val_folder)

        # list moving function
        def list_index_move(list, split_num):
            split_idx = int(round(len(list) * split_num))  # split index
            new_list = list[split_idx:]
            new_list = new_list + list[:split_idx]
            return new_list

        # sc and sf list moving
        sc_list = list_index_move(sc_list, split_num=cfg.params['val_split'])
        sf_list = list_index_move(sf_list, split_num=cfg.params['val_split'])
