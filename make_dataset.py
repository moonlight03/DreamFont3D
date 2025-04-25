import os
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

a1 = []
for i in range(65, 91):
    a1.append(chr(i))
print(a1)
a1.append('A')


ttfs = os.listdir('TTF')
ttfs.remove('FZJSSB.TTF')  # ttf file, download from https://www.foundertype.com/

for ttf in ttfs:
    ttf_name = ttf.split('.')[0]
    ttf = os.path.join('TTF', ttf)
    mask_path = os.path.join('mask_image', ttf_name)
    mask_blur_path = os.path.join('mask_image_blur', ttf_name)
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    if not os.path.exists(mask_blur_path):
        os.mkdir(mask_blur_path)
    fontsize = 196
    font = ImageFont.truetype(ttf, 185, encoding='utf-8')
    # 灰度图
    for i, j in enumerate(a1):
        # RGB图
        image = Image.new('RGB', (fontsize, fontsize))
        a,b,c,d = font.getbbox(j)
        w,h = font.getsize(j)
        draw = ImageDraw.Draw(image)
        draw.text(((fontsize-w)/2, (fontsize-h)/2-b/2), j, font=font)
        image.save(os.path.join(mask_path, '{}-0.png'.format(j)))
        np_image = np.array(image)
        image_blur = cv2.blur(np_image, (10, 10))
        image_blur = Image.fromarray(image_blur)
        image_blur.save(os.path.join(mask_blur_path, '{}-0.png'.format(j)))

        h_side, w_side = d - b, 20
        img = Image.new("RGB", (196, 196), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        a_side, b_side, c_side, d_side = (fontsize - w_side) // 2,  (fontsize - h_side) // 2, (fontsize - w_side) // 2 + w_side, (fontsize - h_side) // 2 + h_side
        draw.rectangle((a_side, b_side, c_side, d_side), fill=(255, 255, 255), outline=(255, 255, 255))

        img.save(os.path.join(mask_path, '{}-90.png'.format(j)))
        img.save(os.path.join(mask_path, '{}-270.png'.format(j)))
        np_img = np.array(img)
        np_img = cv2.blur(np_img, (10, 10))
        image_blur = Image.fromarray(np_img)
        image_blur.save(os.path.join(mask_blur_path, '{}-90.png'.format(j)))
        image_blur.save(os.path.join(mask_blur_path, '{}-270.png'.format(j)))
        rotate = [45, 135, 180, 225, 315]
        for r in rotate:
            left = int((fontsize - w) / 2)
            top = int((fontsize - h) // 2 + b // 2)
            xy1, xy2, xy3, xy4 = [left, top], [left + w, top], [left, top + d - b], [left + w, top + d - b]
            gap = int(w * (1 - math.cos(2*math.pi / 360 * r)) / 2)
            xy1_, xy2_, xy3_, xy4_ = [left + gap, top], [left + w - gap, top], [left + gap, top + d - b], [left + w - gap, top + d - b]

            post1 = np.float32([xy1, xy2, xy3, xy4])
            post2 = np.float32([xy1_, xy2_, xy3_, xy4_])
            M = cv2.getPerspectiveTransform(post1, post2)
            np_image_r = cv2.warpPerspective(np_image, M, (196, 196))
            np_image_r_save = Image.fromarray(np_image_r)
            np_image_r_save.save(os.path.join(mask_path, '{}-{}.png'.format(j, r)))
            np_image_r_blur = cv2.blur(np_image_r, (10, 10))
            np_image_r_blur = Image.fromarray(np_image_r_blur)
            np_image_r_blur.save(os.path.join(mask_blur_path, '{}-{}.png'.format(j, r)))
