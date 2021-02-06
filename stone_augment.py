import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

K_STONE_DIR = 'data/cropped_stone_dataset_new/k_stone_dataset'
K_STONE_DIST_DIR = 'data/cropped_stone_dataset_new/k_stone_dist_map'
U_STONE_DIR = 'data/cropped_stone_dataset_new/u_stone_dataset'
U_STONE_DIST_DIR = 'data/cropped_stone_dataset_new/u_stone_dist_map'
B_STONE_DIR = 'data/cropped_stone_dataset_new/b_stone_dataset'
B_STONE_DIST_DIR = 'data/cropped_stone_dataset_new/b_stone_dist_map'


def stone_embedding(image, mask, stone, stone_dist_map, location):
    import math

    # stone location
    x = location[0]
    y = location[1]

    # cropped region
    sh, sw = stone.shape[:2]
    x_min = math.floor(x - sh/2)
    y_min = math.floor(y - sw/2)
    x_max = math.ceil(x + sh/2)
    y_max = math.ceil(y + sw/2)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    cropped = image[x_min:x_max, y_min:y_max]
    ch, cw = cropped.shape[:2]
    stone = cv2.resize(stone, (cw, ch))
    stone_dist_map = cv2.resize(stone_dist_map, (cw, ch))

    # create stone mask
    stone_mask = np.zeros(stone.shape[:2])
    stone_mask[stone != 0] = 255

    # combine stone and cropped region
    stone_weight = random.randrange(10, 15) / 100
    combine = stone * stone_weight + cropped
    combine[stone_mask == 0] = cropped[stone_mask == 0]

    # gaussian blur based on distance map
    blur = cv2.GaussianBlur(combine, (3, 3), 0)
    stone_dist_map = np.clip((stone_dist_map / 255) * 1.5, 0, 1)
    embedded_stone_region = blur * stone_dist_map + cropped * (1 - stone_dist_map)
    # embedded_stone_region3 = blur3 * stone_dist_map + cropped * (1 - stone_dist_map)
    #
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax1.imshow(cropped, cmap='gray', vmin=0, vmax=255)
    # ax2.imshow(blur, cmap='gray', vmin=0, vmax=255)
    # ax3.imshow(stone_dist_map * 255, cmap='gray', vmin=0, vmax=255)
    # ax4.imshow(embedded_stone_region, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # create new image embedding new stone
    embedded_stone_img = image
    embedded_stone_img[x_min:x_max, y_min:y_max] = embedded_stone_region

    # create new mask
    embedded_stone_mask = mask
    embedded_stone_mask[x_min:x_max, y_min:y_max] = stone_mask

    return embedded_stone_img, embedded_stone_mask


def stone_augmenting_generator(image, mask, stone_location_map, stone_num, prob_map=False):
    import random
    import imutils

    # stone list from cropped stone databased
    k_stone_list = os.listdir(K_STONE_DIR)
    u_stone_list = os.listdir(U_STONE_DIR)
    b_stone_list = os.listdir(B_STONE_DIR)

    # get all location map coordinate list
    location_list = np.argwhere(stone_location_map == 255)  # uniform distribution
    if prob_map:
        location_list = np.argwhere(stone_location_map > 0)  # probability map - indice
        prob_map = stone_location_map[stone_location_map > 0]  # probability map - values

    new_image = image
    new_mask = mask
    for i in range(stone_num):

        # random stone coordinate(x,y,z) in location list
        location = random.choice(location_list)  # uniform distribution
        print(location)
        if prob_map:
            location = random.choices(location_list, weights=prob_map)
            location = location[0]

        # random stone from database  - load stone and its distance map
        if location[2] == 2:  # kidney stone
            stone_name = random.choice(k_stone_list)
            stone = cv2.imread((K_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            stone_dist_map = cv2.imread((K_STONE_DIST_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

        elif location[2] == 1:  # ureter stone
            stone_name = random.choice(u_stone_list)
            stone = cv2.imread((U_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            stone_dist_map = cv2.imread((U_STONE_DIST_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

        elif location[2] == 0:  # bladder region
            stone_name = random.choice(b_stone_list)
            stone = cv2.imread((B_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            stone_dist_map = cv2.imread((B_STONE_DIST_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

        # resize cropped stone
        stone = cv2.resize(stone, (round(stone.shape[0] / 1), round(stone.shape[1] / 1)))
        stone_dist_map = cv2.resize(stone_dist_map, stone.shape[:2])

        # stone rotating
        angle = random.randrange(-5, 5)
        stone = imutils.rotate(stone, angle)  # image
        stone_dist_map = imutils.rotate(stone_dist_map, angle)  # image

        # horizontal flip
        if bool(random.getrandbits(1)):
            stone = cv2.flip(stone, 1)
            stone_dist_map = cv2.flip(stone_dist_map, 1)

        # vertical flip
        if bool(random.getrandbits(1)):
            stone = cv2.flip(stone, 0)
            stone_dist_map = cv2.flip(stone_dist_map, 0)

        # random stone coordinate in location list
        new_image, new_mask = stone_embedding(image, mask, stone, stone_dist_map, location)

    return new_image, new_mask


if __name__ == '__main__':
    import os

    image_name = '15005673'
    # image
    image = cv2.imread(('data/all_images_full/' + image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1024, 1024))

    # mask
    mask = np.zeros(image.shape)
    # mask = cv2.imread(('data/all_groundtruth/' + image_name + '.png'), cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask, (256, 256))

    # stone location map
    stone_location_map = cv2.imread(('data/Full_KUB_map/' + image_name + '.png'))
    stone_location_map = cv2.resize(stone_location_map, (image.shape[0], image.shape[1]))

    stone_num = random.randrange(1, 3)  # random stone number

    new_image, new_mask = stone_augmenting_generator(image, mask, stone_location_map, stone_num)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(new_image, cmap='gray', vmin=0, vmax=255)
    ax2.imshow(new_mask, cmap='gray', vmin=0, vmax=255)
    plt.show()