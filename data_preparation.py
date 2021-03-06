import os
import json
import random

'''
函数作用：在不打乱数据集的情况下，返回训练集数据路径，根据验证集比例因子，随机选取var_rate比例的验证集
'''


def split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), 'dataset root:{} does not exist'.format(root)

    obj_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join
                                                                  (root, cla))]
    obj_class.sort()
    # print('obj_class:', obj_class)

    class_indices = dict((k, v) for v, k in enumerate(obj_class))
    # print("class_indices:", class_indices)

    json_str = json.dumps(dict((k, v) for k, v in enumerate(class_indices)), indent=4)
    # json_str = json.dumps(dict((v, k) for k, v in class_indices.items()), indent=4)
    with open('class_indices.json', 'w')as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cla in obj_class:
        cla_path = os.path.join(root, cla)
        # 获取每个分类下的支持的文件路径
        # splitext:分离文件名与扩展名

        images = []
        for name in os.listdir(cla_path):
            if os.path.splitext(name)[-1] in supported:
                images.append(os.path.join(root, cla, name))

        # images = [os.path.join(root, cla, name) for name in os.listdir(cla_path)
        #           if os.path.splitext(name)[-1] in supported]
        # print(images)

        images_class = class_indices[cla]
        # 记录当前样本数量
        every_class_num.append(len(images))

        # 截取长度为k的随机元素
        val_path = random.sample(images, k=int(len(images) * val_rate))
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(images_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(images_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print('data split done!\n')
    return train_images_path, train_images_label, val_images_path, val_images_label


if __name__ == '__main__':
    data_root = '/home/lee/pyCode/dl_data/flower_photos'
    split_data(data_root)
