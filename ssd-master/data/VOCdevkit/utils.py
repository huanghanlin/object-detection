# coding:utf-8
# TODO: wrap ordinary method for dataset processing
import numpy as np
import random
import pydicom
import cv2
import os
from albumentations.augmentations import functional
#from config_train import train_cf



def left_or_right(img):
    '''
    TODO: judege the breast in left region of image or right
    :param img: input im
    :return: bool flag
    '''
    flag = False    # breast in right region then flag is False
    row,col = img.shape
    #center_row = int(row/2)
    center_col = int(col /2)

    mean_left = np.mean(img[0:row,0:center_col])
    mean_right = np.mean(img[0:row,center_col:col])

    if mean_left > mean_right:
        flag = True
    else:
        flag = False

    return flag


def getCH(img,nbins=256):
    '''
    TODO: get img Cumulative distribution histogram
    :param img: ndarray input image
    :param nbins: integer histogram bins
    :return ch： ndarray result of Cumulative distribution histogram
    '''
    # get image histogram
    imgmax = img.max()
    imgmin = img.min()

    hist,bins = np.histogram(img.flatten(),nbins,[imgmin,imgmax])

    area = img.shape[0]*img.shape[1]
    # calculate cumulative histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf /area   # get normalized cumulative distribution histogram

    return cdf_normalized, bins

def getMajorGrey(nbins,cd_normalized,th):
    '''
    TODO: get img majority object grey intensity distribution
    :param img:
    :param nbins:
    :param cd_normalized:
    :return:
    '''

    th_min = th
    th_max = 1.0 - th

    cd_normalized = np.array(cd_normalized)
    index_th_min = np.where(cd_normalized > th_min)[0][0]
    index_th_max = np.where(cd_normalized > th_max)[0][0]

    major_min = nbins[index_th_min].astype(np.uint16)
    major_max = nbins[index_th_max].astype(np.uint16)


    return major_max, major_min


def cropImage(image):
    cdf_normal, nbins = getCH(image,256)
    major_max,major_min = getMajorGrey(nbins,cdf_normal,0.02)
    image[image >= major_max] = 0
    image[image <= major_min] = 0
    return image

def judgeCrop(img):
    '''
    TODO: judege whether image need crop high gray value region
    :param image:
    :return:
    '''
    imgmax = img.max()
    imgmin = img.min()
    nbins = 256

    hist, bins = np.histogram(img.flatten(), nbins, [imgmin, imgmax])
    cdf = hist.cumsum()




def resizeImage(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = cv2.resize(image,(round(w * scale),round(h * scale),),cv2.INTER_LANCZOS4)
    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    #return image.astype(image_dtype), window, scale, padding, crop
    return image

def imgPreProcess(image):
    '''
    TODO: preprocess for image load as src
    :param image:
    :return:
    '''


'''
def load_img(img_name,tag,root,shape):
    '''导入图像'''
    # 导入mask时，因为mask都是png，所以需要对img_name就行修饰
    if tag.find('mask') != -1:
        if img_name.endswith('.dcm'):
            img_name = img_name.split('.dcm')[0] + '.png'
        else:
            img_name = img_name

    img_path = os.path.join(os.path.join(root,tag),img_name)
    if img_path.endswith('.dcm'):
        dcm = pydicom.read_file(img_path)
        img = resizeImage(dcm.pixel_array,train_cf.train_shape[0],train_cf.train_shape[1])
        #img = cv2.resize(img,(shape[0],shape[1]), cv2.INTER_LANCZOS4)
        img = img.astype(np.float)
    elif img_path.endswith('.png'):
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = resizeImage(img, train_cf.train_shape[0], train_cf.train_shape[1])
        #img = cv2.resize(img, shape, cv2.INTER_LANCZOS4)
        img = img.astype(np.float)
    else:
        raise Exception("{} is not exist".format(img_path))
    if tag.find('mask') != -1:   # 针对mask进行处理
        img[img !=0] = 1

    return img'''

def nonlinearImg(soft, mask):
    '''生成灰度分布上增广，使用拟合多项式'''
    if mask is not None:
        softtemp = soft * mask
    else:
        softtemp = soft

    vmax = softtemp[softtemp > 0].max()
    vmin = softtemp[softtemp > 0].min()

    soft_norm = soft.copy()
    soft_norm[soft_norm > vmax] = vmax
    soft_norm[soft_norm < vmin] = vmin
    soft_norm = (soft_norm - vmin) / (vmax - vmin)

    It = random.randint(5, 20)
    alpha_sum = 0
    v = 0.4
    if random.uniform(0, 1) > 0.5:
        n = random.uniform(v, 1.0 / v)
        alpha = random.uniform(0.01, 1.0)
        newsoft = alpha * np.power(soft_norm, n)
        alpha_sum += alpha
        for i in range(It - 1):
            n = random.uniform(v, 1.0 / v)
            alpha = random.uniform(0.01, 1.0)
            newsoft += alpha * np.power(soft_norm, n)
            alpha_sum += alpha

        newsoft = newsoft / alpha_sum

        newsoft = newsoft * (vmax - vmin) + vmin
        newsoft[soft == 0] = 0
    else:
        newsoft = soft


    return newsoft, (vmin, vmax)

def grayAg(image):
    '''
    TODO: ordinary gray augmentation
    :param image:
    :return:
    '''
    image, vthreshold = nonlinearImg(image,mask=None)

    return image


################ spatial transform ###################
def spacialAg(img1,img2):


    # 水平翻转
    if np.random.random() < 0.5:
        img1 = functional.hflip(img1)
        img2 = functional.hflip(img2)
    # 垂直翻转
    if np.random.random() < 0.5:
        img1 = functional.vflip(img1)
        img2 = functional.vflip(img2)

    # 绕图像中心旋转
    if np.random.random() < 0.5:
        angle = np.random.uniform(-20, 20)
        scale = np.random.uniform(1 - 0.1, 1 + 0.1)
        dx = np.random.uniform(-0.0625, 0.0625)
        dy = np.random.uniform(-0.0625, 0.0625)
        img1 = functional.shift_scale_rotate(img1, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR,
                                             border_mode=cv2.BORDER_CONSTANT)
        img2 = functional.shift_scale_rotate(img2, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR,
                                             border_mode=cv2.BORDER_CONSTANT)

    # 网格扭曲
    if np.random.random() < 0.5:
        num_steps = 5
        distort_limit = (-0.3, 0.3)
        stepsx = [1 + np.random.uniform(distort_limit[0], distort_limit[1]) for i in
                  range(num_steps + 1)]
        stepsy = [1 + np.random.uniform(distort_limit[0], distort_limit[1]) for i in
                  range(num_steps + 1)]

        img1 = functional.grid_distortion(img1, num_steps, stepsx, stepsy, interpolation=cv2.INTER_LINEAR,
                                          border_mode=cv2.BORDER_CONSTANT)
        img2 = functional.grid_distortion(img2, num_steps, stepsx, stepsy, interpolation=cv2.INTER_LINEAR,
                                          border_mode=cv2.BORDER_CONSTANT)

    #  弹性扭曲
    if np.random.random() < 0.5:
        alpha = 1
        sigma = 50
        alpha_affine = 50
        interpolation = cv2.INTER_LINEAR
        random_state = np.random.randint(0, 10000)
        img1 = functional.elastic_transform_fast(img1, alpha, sigma, alpha_affine, interpolation,
                                                 cv2.BORDER_CONSTANT, np.random.RandomState(random_state))
        img2 = functional.elastic_transform_fast(img2, alpha, sigma, alpha_affine, interpolation, cv2.BORDER_CONSTANT,
                                                 np.random.RandomState(random_state))


    return img1, img2

def normalize(img, threshold,mask=None,vthreshold=None):
    '''进行归一化'''
    if mask is not None:    # 当mask不为空时，使用mask进行区域提取
        img_cut = img * mask
    else:
        img_cut = img

    if vthreshold is None:
        img_cut_min = img_cut[img_cut > 0].min()
        img_cut_max = img_cut.max()
    else:
        img_cut_min = vthreshold[0]
        img_cut_max = vthreshold[1]

    # 对图像进行归一化处理
    a = (threshold[1] - threshold[0]) / (img_cut_max - img_cut_min)
    b = (threshold[0] * img_cut_max - img_cut_min * threshold[1]) / (img_cut_max - img_cut_min)

    img_new = img *a + b
    img_new[img_new < threshold[0]] = threshold[0]
    img_new[img_new > threshold[1]] = threshold[1]

    return img_new  # 返回归一化后的img

def getlists(root):
    '''根据指定txt文件获取文件名list'''
    if os.path.exists(root):
        filepath = []
        with open(root,"rt") as fp:
            files = fp.readlines()
            for file in files:
                fileitem = file.split("\n")[0]
                if fileitem.endswith("dcm") or fileitem.endswith("png"):
                    filepath.append(fileitem)
                else:
                    raise Exception("no png or dcm files in directory")

        return filepath
    else:
        raise Exception("The file {} is not exist".format(root))

def dimension_normlize(inputs):
    '''对送入网络的张量进行维度上的规范化'''
    if inputs is None:
        return inputs
    else:
        if len(inputs.shape) < 4:
            inputs = inputs.unsqueeze(1)
        else:
            inputs = inputs
        return inputs




def write_txt(path, savepath):
    lists = os.listdir(path)
    f = open(savepath, 'w')
    for file in lists:
        file = file.split('.dcm')[0] + '.dcm'
        f.write(str(file)+'\n')
    f.close()


if __name__ =="__main__":
    path = r'F:\github\ssd-master\data\VOCdevkit\VOC2007\JPEGImages\*png'
    savepath = r'F:\github\ssd-master\data\VOCdevkit\agumentioan\'
    lists = os.listdir(path)
    for file in lists:
        #dcm = pydicom.read_file(os.path.join(path,file))
        #img = dcm.pixel_array
        img = cv2.imread(os.path.join(path, file),cv2.IMREAD_UNCHANGED)
        img = resizeImage(img,300,300)
        filename = file.split('.dcm')[0] + '.png'
        cv2.imwrite(os.path.join(savepath, filename), img)






