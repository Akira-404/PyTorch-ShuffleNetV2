import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import shufflenet_v2_x1

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.56, 0.406), (0.229, 0.224, 0.225))
         ]
    )

    # 载入测试图片
    test_img_path = './tulip.jpeg'
    assert os.path.exists(test_img_path), "{} 文件不存在".format(test_img_path)

    img = Image.open(test_img_path)
    plt.imshow(img)
    img = data_transform(img)

    # 扩充一个维度[c,h,w]->[n,c,h,w]
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "{} 文件不存在".format(json_path)

    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    model = shufflenet_v2_x1(num_classes=5).to(device)

    # 加载权重
    weights_path = './model_data.pth'
    assert os.path.exists(weights_path), "{} 文件不存在".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    # 模型推理
    model.eval()
    with torch.no_grad():
        # 去除一个维度
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "类别: {}   概率: {:.3}".format(class_indict[str(predict_cla)],
                                            predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
