import numpy as np
from PIL import Image
import torchvision

def showImage(imageArray):
    # print(imageArray)
    img = Image.fromarray(imageArray, 'RGB')
    img.show()

def saveImage(imageArray, name="default"):
    img = Image.fromarray(imageArray, 'RGB')
    img.save('image1.png')

def showTensor(tensor):
    img = torchvision.transforms.ToPILImage()(tensor)
    img.show()

def saveTensor(tensor):
    img = torchvision.transforms.ToPILImage()(tensor)
    img.save('img/tensor1.png')


#TODO doens't work yes, how to mask replace an array equeal to a certain array?
def fancyPreprocess(image):
    for i in range(len(image)):
        mask = np.array([107, 107, 107])
        print(mask)
        image[i][mask] = [255, 0, 255]
    return image

def standardPreprocess(image):
    image = np.mean(image[:, :], axis=2)
    # image = image[:2, :2]
    return image