
from PIL import Image
import os, cv2

PATH = os.getcwd()
data_path = "./spectograms/"
data_dir_list = os.listdir(data_path)

img_rows = 28
img_cols = 28
num_classes = 3
img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        image = Image.open(data_path + '/' + dataset + '/' + img)
        image = image.resize((28, 28), Image.ANTIALIAS)
        image = image.convert('RGB')
        image.save('./resized_spectograms/' + dataset + '/' + img + '.jpg', "JPEG")

