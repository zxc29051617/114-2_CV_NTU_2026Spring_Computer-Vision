import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # 1. 讀 setting file
    with open(args.setting_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # sigma_s, sigma_r
    sigma_info = lines[6].split(',')
    sigma_s = int(sigma_info[1])
    sigma_r = float(sigma_info[3])

    # RGB weights
    weight_list = []
    for line in lines[1:6]:
        r, g, b = map(float, line.split(','))
        weight_list.append((r, g, b))
     # 2. 建立 JBF 物件
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)

    # 3. baseline: bilateral filter
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)

    # 4. 建立 6 張 guidance images
    guidance_images = []
    guidance_names = []

    # grayscale
    for idx, (wr, wg, wb) in enumerate(weight_list):
        guidance_img = wr * img_rgb[:, :, 0] + wg * img_rgb[:, :, 1] + wb * img_rgb[:, :, 2]
        guidance_images.append(guidance_img)
        guidance_names.append(f'grayscale_{idx+1}')
    
    # OpenCV gray
    guidance_images.append(img_gray)
    guidance_names.append('opencv_gray')

    # 5. 每張 guidance 跑 JBF 並算 cost
    costs = []
    outputs = []
    for i in range(len(guidance_images)):
        guidance = guidance_images[i]
        output = JBF.joint_bilateral_filter(img_rgb, guidance)
        cost = np.sum(np.abs(output.astype(np.int32) - bf_out.astype(np.int32)))
        costs.append(cost)
        outputs.append(output)
        print(f'Guidance: {guidance_names[i]}, Cost: {cost}')
    # 6. 顯示結果
    best_idx = np.argmin(costs)
    worst_idx = np.argmax(costs)
    print('Best guidance:', guidance_names[best_idx], 'cost =', costs[best_idx])
    print('Worst guidance:', guidance_names[worst_idx], 'cost =', costs[worst_idx])

    # 7. 存圖
    os.makedirs('output', exist_ok=True)

    cv2.imwrite('output/bf_out.png', cv2.cvtColor(bf_out, cv2.COLOR_RGB2BGR))

    for i in range(len(outputs)):
        cv2.imwrite(f'output/{guidance_names[i]}_guidance.png', guidance_images[i])
        cv2.imwrite(f'output/{guidance_names[i]}_out.png', cv2.cvtColor(outputs[i], cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()