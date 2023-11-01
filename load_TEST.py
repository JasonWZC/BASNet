from PIL import Image
import glob
import torchvision.transforms as transforms
import torch.utils.data
import random
import torchvision.transforms.functional as tf

def read_images():
    imgs_Atrain = glob.glob('./datasets/TEST/train' + '\\A\\*.png')
    imgs_Btrain = glob.glob('./datasets/TEST/train' + '\\B\\*.png')
    labelstrain = glob.glob('./datasets/TEST/train' + '\\label\\*.png')
    imgs_Atest = glob.glob('./datasets/TEST/test' + '\\A\\*.png')
    imgs_Btest = glob.glob('./datasets/TEST/test' + '\\B\\*.png')
    labelstest = glob.glob('./datasets/TEST/test' + '\\label\\*.png')
    return imgs_Atrain, imgs_Btrain, labelstrain, imgs_Atest, imgs_Btest, labelstest

"""
Divide the training set and test set
"""
# def split_dataset(before, after, change):
#     train_size = int(0.8 * len(before))
#     random.seed(train_size)
#     random.shuffle(before)
#     random.seed(train_size)
#     random.shuffle(after)
#     random.seed(train_size)
#     random.shuffle(change)
#
#     random.seed(len(before)-train_size)
#     train_dataset_before = random.sample(before, train_size)
#     random.seed(len(before)-train_size)
#     train_dataset_after = random.sample(after, train_size)
#     random.seed(len(before)-train_size)
#     train_dataset_change = random.sample(change, train_size)
#
#     test_dataset_before = [i for i in before if i not in train_dataset_before]
#     test_dataset_after = [i for i in after if i not in train_dataset_after]
#     test_dataset_change = [i for i in change if i not in train_dataset_change]
#     return train_dataset_before, train_dataset_after, train_dataset_change, test_dataset_before, test_dataset_after, test_dataset_change

def imagestransforms(before, after, change, apply_augmentation=False):
    if apply_augmentation:
        if random.random() > 0.6:
            before = tf.hflip(before)
            after = tf.hflip(after)
            change = tf.hflip(change)
        if random.random() < 0.4:
            before = tf.vflip(before)
            after = tf.vflip(after)
            change = tf.vflip(change)
        angle = transforms.RandomRotation.get_params([-180, 180])
        before = before.rotate(angle)
        after = after.rotate(angle)
        change = change.rotate(angle)

    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

def images_transforms(before, after, change):
    before, after, change = imagestransforms(before, after, change)
    return before, after, change

def images_transforms_(before, after, change):
    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

class BuildingChangeDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        # self.before_list, self.after_list, self.change_list = read_images(root_path)
        self.train_dataset_before, self.train_dataset_after, self.train_dataset_change, self.test_dataset_before, self.test_dataset_after, self.test_dataset_change = read_images()
        if mode == 'train':
            print('The train set loaded ' + str(len(self.train_dataset_before)) + ' pictures')
        elif mode == 'test':
            print('The test set loaded ' + str(len(self.test_dataset_before)) + ' pictures')

    def __getitem__(self, item):
        if self.mode == 'train':
            before = Image.open(self.train_dataset_before[item])
            after = Image.open(self.train_dataset_after[item])
            change = Image.open(self.train_dataset_change[item]).convert('L')
            before, after, change = images_transforms(before, after, change)
            return before, after, change
        elif self.mode == 'test':
            before = Image.open(self.test_dataset_before[item])
            after = Image.open(self.test_dataset_after[item])
            change = Image.open(self.test_dataset_change[item]).convert('L')
            before, after, change = images_transforms_(before, after, change)
            return before, after, change

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset_before)
        elif self.mode == 'test':
            return len(self.test_dataset_before)

if __name__ == '__main__':
    train_data = BuildingChangeDataset(mode='train')
    test_data = BuildingChangeDataset(mode='test')