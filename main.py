import tensorflow as tf
import config_file as cfg
import numpy as np
import cv2
import os
import pandas as pd
from myUnet import UNet
# from pretrained_Unet import get_pretrained_unet
from stone_augment import stone_embedding
from segmentation_evaluate import segmentation_evaluate_full, metrics_compute
from KUB_partitions_function import create_KUB_partitions, combine_KUB_partitions
import util_function as utils


# define directory from config file
IMAGE_DIR = cfg.root_path + os.sep + cfg.dataset_path['image_dir']
MASK_DIR = cfg.root_path + os.sep + cfg.dataset_path['gt_dir']
KUB_MAP_DIR = cfg.root_path + os.sep + cfg.dataset_path['KUB_map_dir']

K_STONE_DIR = cfg.root_path + os.sep + cfg.stones_path['k_stone']
K_STONE_DIST_DIR = cfg.root_path + os.sep + cfg.stones_path['k_stone_dist']
U_STONE_DIR = cfg.root_path + os.sep + cfg.stones_path['u_stone']
U_STONE_DIST_DIR = cfg.root_path + os.sep + cfg.stones_path['u_stone_dist']
B_STONE_DIR = cfg.root_path + os.sep + cfg.stones_path['b_stone']
B_STONE_DIST_DIR = cfg.root_path + os.sep + cfg.stones_path['b_stone_dist']

CROSS_VAL_NUM = 0
SAVE_PATH = cfg.root_path + os.sep \
            + cfg.save_path['save_dir']['root_dir'] + os.sep \
            + cfg.save_path['save_dir']['save_folder'] + os.sep \
            + 'cross_val#' + str(CROSS_VAL_NUM)

# define other common used variables
IMAGE_SIZE = cfg.image_params['image_size']
FULL_IMAGE_SIZE = cfg.image_params['full_image_size']


def load_images(images_list, image_type, input_size, partition=True):

    # load partition images
    if partition:
        if image_type == 0:  # image
            images = np.zeros((len(images_list) * 3, input_size, input_size, 1), np.float32)   # [batch_size, w, h, 1]
            for i, image_name in enumerate(images_list):
                image = cv2.imread(os.path.join(IMAGE_DIR, image_name[0]), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (round(image.shape[1] / (image.shape[0] / FULL_IMAGE_SIZE)), FULL_IMAGE_SIZE))

                # read KUB map
                kub_map = cv2.imread(os.path.join(KUB_MAP_DIR, image_name[0][:-4] + '.png'))
                kub_map = cv2.resize(kub_map, (image.shape[1], image.shape[0]))

                # create 3 partitions
                l_partition, r_partition, b_partition = create_KUB_partitions(image, kub_map)

                l_partition = cv2.resize(utils.normalize_x(l_partition), (input_size, input_size))
                images[3 * i] = l_partition[:, :, np.newaxis]

                r_partition = cv2.resize(utils.normalize_x(r_partition), (input_size, input_size))
                images[3 * i + 1] = r_partition[:, :, np.newaxis]

                b_partition = cv2.resize(utils.normalize_x(b_partition), (input_size, input_size))
                images[3 * i + 2] = b_partition[:, :, np.newaxis]

        elif image_type == 1:  # mask
            images = np.zeros((len(images_list) * 3, input_size, input_size, 2), np.float32)     # [batch_size, w, h, 2]
            for i, image_name in enumerate(images_list):
                if image_name[2] == 1:    # sc
                    image = cv2.imread(os.path.join(MASK_DIR, image_name[1]), cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (round(image.shape[1] / (image.shape[0] / FULL_IMAGE_SIZE)), FULL_IMAGE_SIZE))

                    # read KUB map
                    kub_map = cv2.imread(os.path.join(KUB_MAP_DIR, image_name[0][:-4] + '.png'))
                    kub_map = cv2.resize(kub_map, (image.shape[1], image.shape[0]))

                    # create 3 partitions
                    l_partition, r_partition, b_partition = create_KUB_partitions(image, kub_map)

                    l_partition = cv2.resize(utils.normalize_y(l_partition), (input_size, input_size))
                    images[3 * i, :, :, 0] = l_partition
                    images[3 * i, :, :, 1] = np.ones_like(l_partition)

                    r_partition = cv2.resize(utils.normalize_y(r_partition), (input_size, input_size))
                    images[3 * i + 1, :, :, 0] = r_partition
                    images[3 * i + 1, :, :, 1] = np.ones_like(r_partition)

                    b_partition = cv2.resize(utils.normalize_y(b_partition), (input_size, input_size))
                    images[3 * i + 2, :, :, 0] = b_partition
                    images[3 * i + 2, :, :, 1] = np.ones_like(b_partition)

    # load full images
    else:
        if image_type == 0:  # image
            images = np.zeros((len(images_list), input_size, input_size, 1), np.float32)
            for i, image_name in enumerate(images_list):
                image = cv2.imread(os.path.join(IMAGE_DIR, image_name[0]), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (input_size, input_size))
                image = image[:, :, np.newaxis]
                images[i] = utils.normalize_x(image)

        elif image_type == 1:  # mask
            images = np.zeros((len(images_list), input_size, input_size, 2), np.float32)
            for i, image_name in enumerate(images_list):
                if image_name[2] == 1:    # sc
                    image = cv2.imread(os.path.join(MASK_DIR, image_name[1]), cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (input_size, input_size))
                    image = image[:, :, np.newaxis]
                    images[i] = utils.normalize_y(image)

    return images


def stone_augmenting_generator(image, mask, stone_location_map, stone_num, prob_map=False):
    import random
    import imutils

    # stone list from cropped stone databased
    k_stone_list = os.listdir(K_STONE_DIR)
    u_stone_list = os.listdir(U_STONE_DIR)
    b_stone_list = os.listdir(B_STONE_DIR)

    new_image = image
    new_mask = mask
    for i in range(stone_num):

        location_map = np.zeros_like(stone_location_map)

        # get all location map coordinate list
        if prob_map:
            region_select = np.random.choice([2, 1, 0], p=[0.3, 0.3, 0.4])  # k, u, b
            if region_select == 2:
                location_map[:, :, 2] = stone_location_map[:, :, 2]
            if region_select == 1:
                location_map[:, :, 1] = stone_location_map[:, :, 1]
            if region_select == 0:
                location_map[:, :, 0] = stone_location_map[:, :, 0]
        else:
            location_map = stone_location_map

        # random stone coordinate(x,y) in location list : uniform distribution
        location = random.choice(np.argwhere(location_map == 255))

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


def image_mask_augmentation(image, mask, stone_location_map, gen_methods, img_name, image_type, epoch):
    import random
    import imutils

    augmented_image = image
    augmented_mask = mask

    # stone embedding
    if gen_methods['stone_augment']:
        if image_type == 0:    # only SF
            stone_num = random.randint(gen_methods['num_of_stones'][0],
                                       gen_methods['num_of_stones'][1])  # random stone number
            if stone_num > 0:
                save = True
                augmented_image, augmented_mask = stone_augmenting_generator(augmented_image,
                                                                             augmented_mask,
                                                                             stone_location_map,
                                                                             stone_num,
                                                                             prob_map=False)

    # image rotation
    if gen_methods['rotation_range']:
        angle = random.randrange(-gen_methods['rotation_range'], gen_methods['rotation_range'])
        augmented_image = imutils.rotate(image, angle)  # image
        augmented_mask = imutils.rotate(mask, angle)  # mask
        stone_location_map = imutils.rotate(stone_location_map, angle)  # KUB map

    # image horizontal flip
    if gen_methods['horizontal_flip']:
        if bool(random.getrandbits(1)):  # random flip/not flip
            augmented_image = cv2.flip(augmented_image, 1)
            augmented_mask = cv2.flip(augmented_mask, 1)
            stone_location_map = cv2.flip(stone_location_map, 1)  # KUB map

    # image vertical flip
    if gen_methods['vertical_flip']:
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

    if gen_methods['zoom_range']:
        zoom_value = random.randrange(100 - gen_methods['zoom_range'], 100) / 100  # 0.xx -1.00 times
        augmented_image = image_zooming(augmented_image, zoom_value)
        augmented_mask = image_zooming(augmented_mask, zoom_value)
        stone_location_map = image_zooming(stone_location_map, zoom_value)  # KUB map

    # intensty shift ** implement only image uint8 value
    def intensity_shift_funct(img, brightness_range):
        value = random.randrange(-brightness_range, brightness_range) / 255
        img = img + value
        return np.clip(img, 0, 255)

    if gen_methods['brightness_shift']:
        augmented_image = intensity_shift_funct(augmented_image, brightness_range=gen_methods['brightness_shift'])

    # image contrast change  ** implement only image - percentage value
    def contrast_adjust(img, contrast_range):
        alpha = random.randrange(100 - contrast_range, 100 + contrast_range) / 100
        return img * alpha

    if gen_methods['contrast_change']:
        augmented_image = contrast_adjust(augmented_image, contrast_range=gen_methods['contrast_change'])

    # resize image,gt
    # augmented_image = cv2.resize(augmented_image, (IMAGE_SIZE, IMAGE_SIZE))
    # augmented_mask = cv2.resize(augmented_mask, (IMAGE_SIZE, IMAGE_SIZE))

    return augmented_image, augmented_mask, stone_location_map


def generator(sc_samples, sf_samples, batch_size, datagen_methods, shuffle_data, image_size, partition):
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

        sf_samples, select_sf = list_index_move(sf_samples, split_num=cfg.training_params['sf_ratio'])

        # combine sc + selected sf
        samples = sc_samples + select_sf if cfg.training_params['sf_ratio'] > 0 else sc_samples

        # random sample[images,gt] lists
        if shuffle_data:
            random.shuffle(samples)

        epoch += 1
        batch_num = 0
        num_samples = len(samples)

        # for each batch
        for offset in range(0, num_samples, batch_size):
            # Get the samples using in this batch
            batch_samples = samples[offset:offset + batch_size]

            # Initialize x_train and y_train arrays for this batch
            if partition:
                x_train = np.zeros((batch_size*3, image_size, image_size, 1), np.float32)
                y_train = np.ones((batch_size*3, image_size, image_size, 2), np.float32)
            else:
                x_train = np.zeros((batch_size, image_size, image_size, 1), np.float32)
                y_train = np.ones((batch_size, image_size, image_size, 2), np.float32)

            img_type_flag = np.zeros((batch_size*3, 1))
            # For each sample in batch
            for i, batch_sample in enumerate(batch_samples):

                # load image (x) and gt mask (y)
                img_name = batch_sample[0]
                gt_name = batch_sample[1]
                img_type = batch_sample[2]
                img_type_flag[3*i:3*i+2] = img_type

                # read full images
                img = cv2.imread(os.path.join(IMAGE_DIR, img_name), cv2.IMREAD_GRAYSCALE)
                if img_type == 0:      # sf
                    gt = np.zeros(img.shape)

                elif img_type == 1:    # sc
                    gt = cv2.imread(os.path.join(MASK_DIR, gt_name), cv2.IMREAD_GRAYSCALE)

                # resize image,gt for having same size -> prepare for augmentation
                img = cv2.resize(img, (round(img.shape[1]/(img.shape[0]/FULL_IMAGE_SIZE)), FULL_IMAGE_SIZE))
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]))

                # read KUB map
                kub_map = cv2.imread(os.path.join(KUB_MAP_DIR, img_name[:-4] + '.png'))
                kub_map = cv2.resize(kub_map, (img.shape[1], img.shape[0]))

                # ** image augmentation : augment image and gt together
                img, gt, kub_map = image_mask_augmentation(img,
                                                           gt,
                                                           kub_map,
                                                           datagen_methods,
                                                           img_name,
                                                           img_type,
                                                           epoch)

                # partitions
                if partition:
                    # create 3 partitions (org + gt)
                    if cfg.image_params['partition_method'] == 'kub_map':
                        l_img, r_img, b_img = create_KUB_partitions(img, kub_map)
                        l_gt, r_gt, b_gt = create_KUB_partitions(gt, kub_map)

                    # L partition
                    l_img = cv2.resize(utils.normalize_x(l_img), (IMAGE_SIZE, IMAGE_SIZE))  # [-1,1]
                    # x_train[3 * i] = np.repeat(l_img[:, :, np.newaxis], 3, axis=-1)
                    x_train[3 * i] = l_img[:, :, np.newaxis]
                    l_gt = cv2.resize(utils.normalize_y(l_gt), (IMAGE_SIZE, IMAGE_SIZE))    # [0,1]
                    y_train[3 * i, :, :, 0] = l_gt

                    # R partition
                    r_img = cv2.resize(utils.normalize_x(r_img), (IMAGE_SIZE, IMAGE_SIZE))  # [-1,1]
                    # x_train[3 * i + 1] = np.repeat(r_img[:, :, np.newaxis], 3, axis=-1)
                    x_train[3 * i + 1] = r_img[:, :, np.newaxis]
                    r_gt = cv2.resize(utils.normalize_y(r_gt), (IMAGE_SIZE, IMAGE_SIZE))    # [0,1]
                    y_train[3 * i + 1, :, :, 0] = r_gt

                    # B partition
                    b_img = cv2.resize(utils.normalize_x(b_img), (IMAGE_SIZE, IMAGE_SIZE))  # [-1,1]
                    # x_train[3 * i + 2] = np.repeat(b_img[:, :, np.newaxis], 3, axis=-1)
                    x_train[3 * i + 2] = b_img[:, :, np.newaxis]
                    b_gt = cv2.resize(utils.normalize_y(b_gt), (IMAGE_SIZE, IMAGE_SIZE))  # [0,1]
                    y_train[3 * i + 2, :, :, 0] = b_gt

                    # iwcc image-wise
                    if cfg.training_params['iw_contour'] and not cfg.training_params['iw_contour_batch']:
                        if img_type == 1:   # sc
                            l_ccmap = utils.cc2weight(l_gt)
                            y_train[3 * i, :, :, 1] = l_ccmap
                            r_ccmap = utils.cc2weight(r_gt)
                            y_train[3 * i + 1, :, :, 1] = r_ccmap
                            b_ccmap = utils.cc2weight(b_gt)
                            y_train[3 * i + 2, :, :, 1] = b_ccmap
                        elif img_type == 0:  # sf
                            #l_ccmap = utils.cc2weight(l_gt)
                            l_ccmap = np.ones_like(l_gt)
                            y_train[3 * i, :, :, 1] = l_ccmap
                            # r_ccmap = utils.cc2weight(r_gt)
                            r_ccmap = np.ones_like(r_gt)
                            y_train[3 * i + 1, :, :, 1] = r_ccmap
                            # b_ccmap = utils.cc2weight(b_gt)
                            b_ccmap = np.ones_like(b_gt)
                            y_train[3 * i + 2, :, :, 1] = b_ccmap

                    if epoch == 1 and batch_num < 10 and cfg.exp_params['save_augment']:
                        cv2.imwrite(SAVE_PATH + '/partition/' + img_name[:-4] + '_L.png', utils.denormalize_x(l_img))
                        cv2.imwrite(SAVE_PATH + '/partition_gt/' + img_name[:-4] + '_L.png', utils.denormalize_y(l_gt))
                        cv2.imwrite(SAVE_PATH + '/partition/' + img_name[:-4] + '_R.png', utils.denormalize_x(r_img))
                        cv2.imwrite(SAVE_PATH + '/partition_gt/' + img_name[:-4] + '_R.png', utils.denormalize_y(r_gt))
                        cv2.imwrite(SAVE_PATH + '/partition/' + img_name[:-4] + '_B.png', utils.denormalize_x(b_img))
                        cv2.imwrite(SAVE_PATH + '/partition_gt/' + img_name[:-4] + '_B.png', utils.denormalize_y(b_gt))
                        if not cfg.training_params['iw_contour_batch']:
                            cv2.imwrite(SAVE_PATH + '/partition_ccmap/' + img_name[:-4] + '_L.png', utils.normalize(l_ccmap))
                            cv2.imwrite(SAVE_PATH + '/partition_ccmap/' + img_name[:-4] + '_R.png', utils.normalize(r_ccmap))
                            cv2.imwrite(SAVE_PATH + '/partition_ccmap/' + img_name[:-4] + '_B.png', utils.normalize(b_ccmap))

                # full image
                else:
                    # normalize img+gt, add 1 axis, and add example to arrays
                    img = cv2.resize(utils.normalize_x(img), (IMAGE_SIZE, IMAGE_SIZE))  # [-1,1]
                    x_train[i] = img[:, :, np.newaxis]
                    gt = cv2.resize(utils.normalize_y(gt), (IMAGE_SIZE, IMAGE_SIZE))  # [0,1]
                    y_train[i, :, :, 0] = gt

                    # iwcc image-wise
                    if cfg.training_params['iw_contour'] and not cfg.training_params['iw_contour_batch']:
                        if img_type == 1:   # sc
                            ccmap = utils.cc2weight(gt)
                            y_train[i, :, :, 1] = ccmap
                        elif img_type == 0:  # sf
                            ccmap = utils.cc2weight(gt)
                            y_train[i, :, :, 1] = ccmap
                        if epoch == 1 and batch_num < 10 and cfg.exp_params['save_augment']:
                            cv2.imwrite(SAVE_PATH + '/full_ccmap/' + img_name[:-4] + '_L.png', utils.normalize(ccmap))

            # iwcc batch-wise
            if partition and cfg.training_params['iw_contour_batch']:
                y_train[:, :, :, 1] = utils.cc2weight_batch(y_train[:, :, :, 0], img_type_flag)

                # save iwcc
                if epoch == 1 and batch_num < 10 and cfg.exp_params['save_augment']:
                    for b in range(batch_size):
                        cv2.imwrite(SAVE_PATH + '/partition_ccmap/' + str(batch_num) + '_' + str(b) + '_L.png',
                                    utils.normalize(y_train[3 * b, :, :, 1]))
                        cv2.imwrite(SAVE_PATH + '/partition_ccmap/' + str(batch_num) + '_' + str(b) + '_R.png',
                                    utils.normalize(y_train[3 * b + 1, :, :, 1]))
                        cv2.imwrite(SAVE_PATH + '/partition_ccmap/' + str(batch_num) + '_' + str(b) + '_B.png',
                                    utils.normalize(y_train[3 * b + 2, :, :, 1]))

            elif not partition and cfg.training_params['iw_contour_batch']:
                y_train[:, :, :, 1] = utils.cc2weight_batch(y_train[:, :, :, 0], img_type_flag)

                # save iwcc
                if epoch == 1 and batch_num < 10 and cfg.exp_params['save_augment']:
                    for b in range(batch_size):
                        cv2.imwrite(SAVE_PATH + '/full_ccmap/' + str(batch_num) + '_' + str(b) + '.png',
                                    utils.normalize(y_train[b, :, :, 1]))

            batch_num += 1

            # yield the next training batch
            yield x_train, y_train


def train(train_samples, val_samples):

    sc_train_samples = train_samples[0]
    sf_train_samples = train_samples[1]

    # define data augmentation methods
    datagen = cfg.datagen

    # batch size (partition : batch_size/3)
    batch_size = round(cfg.training_params['batch_size']/3) if cfg.image_params['partition'] else cfg.training_params['batch_size']

    # create a generator object for training samples
    train_generator = generator(sc_train_samples,
                                sf_train_samples,
                                batch_size=batch_size,
                                datagen_methods=datagen,
                                shuffle_data=True,
                                image_size=IMAGE_SIZE,
                                partition=cfg.image_params['partition'])

    # load validating samples  // image_type: 0-image, 1-gt
    x_val = load_images(val_samples,
                        image_type=0,
                        input_size=IMAGE_SIZE,
                        partition=cfg.image_params['partition'])

    y_val = load_images(val_samples,
                        image_type=1,
                        input_size=IMAGE_SIZE,
                        partition=cfg.image_params['partition'])

    print('#train sc = ' + str(3 * len(sc_train_samples) if cfg.image_params['partition'] else len(sc_train_samples)))
    print('#train sf = ' + str(3 * len(sf_train_samples) if cfg.image_params['partition'] else len(sf_train_samples)))

    # load U-Net model
    print('Load model ...')
    network = UNet(cfg.model_params['input_channel'],
                   cfg.model_params['output_channel'],
                   cfg.model_params['first_layer_filters'])

    model = network.get_model()
    model.summary()

    # define loss, optimizer, and monitoring metrics to model
    model.compile(loss=cfg.training_params['loss_function'],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.training_params['initial_lr']),
                  metrics=cfg.training_params['evaluate_methods'])

    # define early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='min',
                                          verbose=1,
                                          min_delta=0.001,
                                          patience=80)

    # define learning rate decay function
    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=10,
                                                              min_lr=5e-5)

    # model checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=SAVE_PATH + '/weight_checkpoint.hdf5',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=SAVE_PATH + '/logs',
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)

    # fits the model on batches with real-time data augmentation:
    total_samples = len(sc_train_samples) + round(len(sf_train_samples)*cfg.training_params['sf_ratio'])
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=int(total_samples / batch_size),
                                  epochs=cfg.training_params['epoch'],
                                  callbacks=[model_checkpoint_callback, es, lr_reduce_callback],
                                  validation_data=(x_val, y_val))

    model.save_weights(SAVE_PATH + '/model_weights.hdf5')

    # save graphs
    if cfg.exp_params['save_graphs']:
        utils.plot_summary_graph(history, SAVE_PATH)

    # save all values from training process to xlsx
    if cfg.exp_params['save_train_history']:
        loss_hist = pd.DataFrame(history.history)
        loss_hist.to_excel(SAVE_PATH + '/history.xlsx')


# Predict Result
def predict(test_samples):
    import cv2

    # testData
    x_test = load_images(test_samples,
                         image_type=0,
                         input_size=IMAGE_SIZE,
                         partition=cfg.image_params['partition'])

    # define U-Net parameters
    network = UNet(cfg.model_params['input_channel'],
                   cfg.model_params['output_channel'],
                   cfg.model_params['first_layer_filters'])
    model = network.get_model()
    # model = get_pretrained_unet(IMAGE_SIZE, IMAGE_SIZE, 3, 1)

    model.load_weights(SAVE_PATH + '/unet_weights.hdf5')

    # predict
    y_pred = model.predict(x_test, batch_size=16)

    # combine partitioned y_pred into full y_pred
    if cfg.image_params['partition']:
        n = 0
        y_pred = y_pred[:, :, :, 0]
        full_y_pred = np.zeros((round(len(y_pred)/3), FULL_IMAGE_SIZE, FULL_IMAGE_SIZE), np.float32)
        for offset in range(0, len(y_pred), 3):
            partitions = y_pred[offset:offset + 3]
            full_kub_map = cv2.resize(cv2.imread(KUB_MAP_DIR + os.sep + test_samples[n][0][:- 4] + '.png'),
                                      (FULL_IMAGE_SIZE, FULL_IMAGE_SIZE))
            full_y_pred[n] = combine_KUB_partitions(partitions[0], partitions[1], partitions[2], full_kub_map)
            n += 1
    else:
        full_y_pred = y_pred[:, :, :, 0]

    return full_y_pred


def evaluate_results(test_samples, all_y_pred, k_fold_num):

    all_y_true = load_images(test_samples,
                             image_type=1,
                             input_size=IMAGE_SIZE,
                             partition=False)

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
                     'stone_type': [],
                     'detect': []}

    sc_num = 0     # for counting sc images
    for i, y_pred in enumerate(all_y_pred):

        image_name = test_samples[i][0][:- 4]

        # read org image
        img = cv2.imread(IMAGE_DIR + os.sep + test_samples[i][0])
        img = cv2.resize(img, (FULL_IMAGE_SIZE, FULL_IMAGE_SIZE))    # 1024 * 1024

        # read KUB map
        kub_map = cv2.imread(KUB_MAP_DIR + os.sep + test_samples[i][0][:-4] + '.png')
        kub_map = cv2.resize(kub_map, (FULL_IMAGE_SIZE, FULL_IMAGE_SIZE))    # 1024 * 1024

        # predicted image
        y_pred = cv2.resize(y_pred, (img.shape[1], img.shape[0]))
        heatmap = np.maximum(y_pred, 0)
        y_pred = utils.denormalize_y(y_pred)
        y_pred = np.array(y_pred, dtype=np.uint8)

        # gt image
        y_true = cv2.resize(all_y_true[i, :, :, 0], (img.shape[1], img.shape[0]))
        y_true = utils.denormalize_y(y_true)
        y_true = np.array(y_true, dtype=np.uint8)

        # evaluation function
        evaluate_pixelbased, evaluate_regionbased, stone_data = segmentation_evaluate_full(y_pred, y_true, kub_map)

        # collecting results
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
            stone_results['stone_type'].append(stone_data[i][6])
            stone_results['detect'].append(stone_data[i][7])

        # heat map generate
        heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        hif = .8
        y_pred_superimposed = heatmap * hif + img

        # write predicted image
        if cfg.exp_params['save_results']:
            if test_samples[i][2] == 1:      #sc
                sc_num = sc_num + 1
                cv2.imwrite(SAVE_PATH + '/PredictedResults/sc/' + image_name + '_Result.png', y_pred)
                cv2.imwrite(SAVE_PATH + '/PredictedResults/sc_heatmap/' + image_name + '_Result.png', y_pred_superimposed)

            elif test_samples[i][2] == 0:    #sf
                cv2.imwrite(SAVE_PATH + '/PredictedResults/sf/' + image_name + '_Result.png', y_pred)
                cv2.imwrite(SAVE_PATH + '/PredictedResults/sf_heatmap/' + image_name + '_Result.png', y_pred_superimposed)

    if cfg.exp_params['save_excel']:
        # save pixel-based results
        excel_pixel = pd.DataFrame(evaluation_results_pixel, columns=['image_name',
                                                                      'total_stones',
                                                                      'TP', 'FN', 'FP',
                                                                      'recall', 'precision', 'F1', 'F2'])
        # excel_pixel.to_excel(SAVE_PATH + '/evaluation_pixelbased.xlsx', index=None, header=True)

        # save region-based results
        excel_region = pd.DataFrame(evaluation_results, columns=['image_name',
                                                                 'total_stones',
                                                                 'TP', 'FN', 'FP',
                                                                 'recall', 'precision', 'F1', 'F2'])
        # excel_region.to_excel(SAVE_PATH + '/evaluation_regionbased.xlsx', index=None, header=True)

        # save stone results
        excel_stone = pd.DataFrame(stone_results, columns=['stone_name',
                                                           '(x,y)', '(w,h)',
                                                           'stone_size', 'stone_type', 'detect'])
        excel_stone.to_excel(SAVE_PATH + '/stone_results(' + str(k_fold_num+1) +').xlsx', index=None, header=True)


if __name__ == '__main__':
    import random
    print(tf.__version__)

    # path
    folder_path = cfg.root_path + os.sep \
                  + cfg.save_path['save_dir']['root_dir'] + os.sep \
                  + cfg.save_path['save_dir']['save_folder']

    sc_list = pd.DataFrame(pd.read_excel(folder_path + '/sc_data.xlsx', engine='openpyxl'), columns=['image', 'gt', 'stone']).values.tolist()
    sf_list = pd.DataFrame(pd.read_excel(folder_path + '/sf_data.xlsx', engine='openpyxl'), columns=['image', 'gt', 'stone']).values.tolist()

    # experiment methods (k-fold cross validation / simple train-test)
    if cfg.exp_params['exp_method'] == 'cross_val':
        num_cross_val = int(1/cfg.exp_params['test_split'])
        for i in range(num_cross_val):

            # create folders in working directory
            CROSS_VAL_NUM = i
            print('cross validation #' + str(CROSS_VAL_NUM+1))
            cross_val_folder = 'cross_val#' + str(CROSS_VAL_NUM+1)

            SAVE_PATH = cfg.root_path + os.sep \
                        + cfg.save_path['save_dir']['root_dir'] + os.sep \
                        + cfg.save_path['save_dir']['save_folder'] + os.sep \
                        + cross_val_folder

            try:
                os.mkdir(folder_path + os.sep + cross_val_folder)
                if cfg.image_params['partition']:
                    os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'partition')
                    os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'partition_gt')
                    os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'partition_ccmap')
                else:
                    os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'full_ccmap')
                os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults')
                os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sc')
                os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sc_heatmap')
                os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sf')
                os.mkdir(folder_path + os.sep + cross_val_folder + os.sep + 'PredictedResults' + os.sep + 'sf_heatmap')
            except FileExistsError:
                print('folder exist')

            # train/test split
            sc_train_samples, sc_test_samples = utils.split_train_test(sc_list, split_ratio=cfg.exp_params['test_split'])  #sc
            sf_train_samples, sf_test_samples = utils.split_train_test(sf_list, split_ratio=cfg.exp_params['test_split'])  #sf

            print('#train = ' + str(len(sc_train_samples + sf_train_samples)))
            print('#test = ' + str(len(sc_test_samples + sf_test_samples)))

            # training U-net  (validation set in training is only sc dataset)
            print('########## Training ###########')
            train([sc_train_samples, sf_train_samples], sc_test_samples)

            # testing
            print('########## Testing ###########')
            y_pred = predict(sc_test_samples + sf_test_samples)

            # evaluate
            print('######### Evaluation ###########')
            evaluate_results(sc_test_samples + sf_test_samples, y_pred, i)

            # sc and sf list moving
            sc_list = utils.list_index_move(sc_list, split_num=cfg.exp_params['test_split'])
            sf_list = utils.list_index_move(sf_list, split_num=cfg.exp_params['test_split'])

    # simple train/val/test
    else:
        # create folders in working directory
        print('normal train/test split')

        SAVE_PATH = cfg.root_path + os.sep \
                    + cfg.save_path['save_dir']['root_dir'] + os.sep \
                    + cfg.save_path['save_dir']['save_folder']

        try:
            if cfg.image_params['partition']:
                os.mkdir(folder_path + os.sep + 'partition')
                os.mkdir(folder_path + os.sep + 'partition_gt')
                os.mkdir(folder_path + os.sep + 'partition_ccmap')
            else:
                os.mkdir(folder_path + os.sep + 'full_ccmap')
            os.mkdir(folder_path + os.sep + 'PredictedResults')
            os.mkdir(folder_path + os.sep + 'PredictedResults' + os.sep + 'sc')
            os.mkdir(folder_path + os.sep + 'PredictedResults' + os.sep + 'sc_heatmap')
            os.mkdir(folder_path + os.sep + 'PredictedResults' + os.sep + 'sf')
            os.mkdir(folder_path + os.sep + 'PredictedResults' + os.sep + 'sf_heatmap')
        except FileExistsError:
            print('folder exist')

        # train+val/test split
        sc_train_val_samples, sc_test_samples = utils.split_train_test(sc_list, split_ratio=cfg.exp_params['test_split'])  # sc
        sf_train_val_samples, sf_test_samples = utils.split_train_test(sf_list, split_ratio=cfg.exp_params['test_split'])  # sf

        # train/val split
        sc_train_samples, sc_val_samples = utils.split_train_test(sc_train_val_samples, split_ratio=cfg.exp_params['val_split'])  # sc
        sf_train_samples, sf_val_samples = utils.split_train_test(sf_train_val_samples, split_ratio=cfg.exp_params['val_split'])  # sf

        print('#train = ' + str(len(sc_train_samples + sf_train_samples)))
        print('#val = ' + str(len(sc_val_samples + sf_val_samples)))
        print('#test = ' + str(len(sc_test_samples + sf_test_samples)))

        # training U-net  (validation set in training is only sc dataset)
        print('########## Training ###########')
        train([sc_train_samples, sf_train_samples], sc_val_samples)

        # testing
        print('########## Testing ###########')
        y_pred = predict(sc_test_samples + sf_test_samples)

        # evaluate
        print('######### Evaluation ###########')
        evaluate_results(sc_test_samples + sf_test_samples, y_pred)
