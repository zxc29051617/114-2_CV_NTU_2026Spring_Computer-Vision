import os
import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis=2), 3, axis=2)
    img = img.astype(np.uint8)

    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    cv2.imwrite(save_path, img)


def normalize_dog_image(dog):
    dog_min = np.min(dog)
    dog_max = np.max(dog)

    if dog_max == dog_min:
        return np.zeros_like(dog, dtype=np.uint8)

    dog_norm = (dog - dog_min) / (dog_max - dog_min)
    dog_norm = (dog_norm * 255).astype(np.uint8)
    return dog_norm


def build_dog_images(img):
    """
    Rebuild Gaussian pyramid and DoG pyramid in main.py
    so we don't need to modify DoG.py
    """
    sigma = 2 ** (1 / 4)
    num_octaves = 2
    num_DoG_images_per_octave = 4
    num_guassian_images_per_octave = num_DoG_images_per_octave + 1

    img = img.astype(np.float64)

    # Step 1: Gaussian pyramid
    gaussian_images = []
    current_base = img.copy()

    for octave in range(num_octaves):
        octave_gaussians = []
        octave_gaussians.append(current_base)  # G0 = base image

        for k in range(1, num_guassian_images_per_octave):
            blurred = cv2.GaussianBlur(
                current_base,
                ksize=(0, 0),
                sigmaX=sigma ** k
            )
            octave_gaussians.append(blurred)

        gaussian_images.append(octave_gaussians)

        if octave != num_octaves - 1:
            last_img = octave_gaussians[-1]
            h, w = last_img.shape
            current_base = cv2.resize(
                last_img,
                (w // 2, h // 2),
                interpolation=cv2.INTER_NEAREST
            )

    # Step 2: DoG pyramid
    dog_images = []
    for octave in range(num_octaves):
        octave_dogs = []
        for i in range(num_DoG_images_per_octave):
            dog = cv2.subtract(
                gaussian_images[octave][i + 1],
                gaussian_images[octave][i]
            )
            octave_dogs.append(dog)
        dog_images.append(octave_dogs)

    return dog_images


def save_dog_images(img, output_dir):
    dog_images = build_dog_images(img)

    for octave in range(2):
        for i in range(4):
            dog_vis = normalize_dog_image(dog_images[octave][i])
            filename = f'DoG{octave + 1}-{i + 1}.png'
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, dog_vis)
            print(f'Saved {save_path}')


def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--mode', default='keypoints', choices=['keypoints', 'dog'], help='visualization mode')
    args = parser.parse_args()

    print('Processing %s ...' % args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    os.makedirs('./output', exist_ok=True)

    if args.mode == 'keypoints':
        DoG = Difference_of_Gaussian(args.threshold)
        keypoints = DoG.get_keypoints(img)

        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        save_path = f'./output/{image_name}_threshold_{args.threshold}.png'
        plot_keypoints(img, keypoints, save_path)

        print(f'Number of keypoints: {len(keypoints)}')
        print(f'Result saved to {save_path}')

    elif args.mode == 'dog':
        save_dog_images(img, './output')


if __name__ == '__main__':
    main()