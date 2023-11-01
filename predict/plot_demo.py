# Get prediction chart
import os
os.chdir('../')
import glob
import cv2
from PIL import Image
from torchvision import transforms
from utils import *
import warnings
warnings.filterwarnings('ignore')

from model.BASNet import BASNet

def predict_on_image(model):
    # pre-processing on image
    to_tensor = transforms.ToTensor()

    file_path = './predict'

    dir = str("BASNet")

    if not os.path.exists(f'{file_path}/{dir}'):
        os.makedirs(f'{file_path}/{dir}')

    val_path_img1 = os.path.join('./predict/TEST', 'A')
    val_path_img2 = os.path.join('./predict/TEST', 'B')

    img1_list = glob.glob(os.path.join(val_path_img1, '*.png'))  # glob.glob()返回一个某一种文件夹下面的某一类型文件路径列表
    img2_list = glob.glob(os.path.join(val_path_img2, '*.png'))

    for i in range(len(img1_list)):

        img1 = Image.open(img1_list[i])
        img2 = Image.open(img2_list[i])

        img1 = to_tensor(img1).float().unsqueeze(0)
        img2 = to_tensor(img2).float().unsqueeze(0)

        # read csv label path
        label_info = get_label_info()
        # predict
        model.eval()
        predict = model(img1, img2).squeeze()
        predict = reverse_one_hot(predict)
        predict = colour_code_segmentation(np.array(predict), label_info)
        predict = cv2.resize(np.uint8(predict), (256, 256))
        cv2.imwrite(f'{file_path}/{dir}/'+"{:}".format(img1_list[i].split('\\')[-1]), cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    print('load model  ...')

    model = BASNet(pretrained=True, normal_init=True)
    model_path = './summaryTEST/BASNet/F1_0.9069_epoch_198.pth'

    try:
        pretrained_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(pretrained_dict)
        print("single GPU!")
    except:
        def load_GPUS(model, model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
            return model
        model = load_GPUS(model, model_path=model_path)
        print("multi GPUs!")

    predict_on_image(model)
    print('Done!')
