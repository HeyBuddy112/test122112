import torch
from torchvision import datasets, transforms
from VAE_model import VAE, loss_function
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



def show_image_tensor(input_tensor: torch.Tensor, save_pic=False, filename=''):
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    pic = toPIL(input_tensor)
    plt.imshow(pic)
    plt.show()
    if save_pic:
        pic.save(filename)

def random_sample_test(model):
    with torch.no_grad():
        z = torch.randn(64, 20)
        image = []
        for i in range(64):   # 随机采样，使用模型解码
            image.append(model.decode(z[i]))
        image = torch.stack(image).view(64, 1, 28, 28)
        final_img = torch.zeros(28 * 8, 28 * 8)
        for i in range(8):
            for j in range(8):  # 将解码后的图片依此添加到生成图像中
                final_img[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = image[8 * i + j][0]
        show_image_tensor(final_img, save_pic=True, filename='saved_img/random_sample_test.jpg')

def normal_sample_test(model):
    with torch.no_grad():
        final_img = torch.zeros(28 * 8, 28 * 8)
        for i in range(8):
            for j in range(8):
                final_img[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = \
                    model.decode(torch.normal(i, j, (1, 20))).view(28, 28)
        show_image_tensor(final_img, save_pic=True, filename='saved_img/normal_sample_test.jpg')


if __name__ == '__main__':
    model = VAE(x_dim=784, h_dim=400, z_dim=20)
    model.load_state_dict(torch.load('checkpoints/checkpoint_150'))

    random_sample_test(model)
    normal_sample_test(model)



