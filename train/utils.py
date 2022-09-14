import cv2
import yaml
import torch
import random
import scipy.stats

import numpy as np
import pandas as pd
import albumentations as A

from albumentations import random_utils

# ----------------------------------------------------------------------------------------------------
# sigmoid for degradation / noise score mapping ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def sigmoid(x, x0, k, b):
    y = b - b/(1 + np.exp(-k*(x-x0)))
    return y


# ----------------------------------------------------------------------------------------------------
# image manipulations --------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def central_crop(image:np.array, width:int, height:int):

    h, w, c = image.shape
    x_margin, y_margin = (h - height)//2, (w - width)//2
    
    return image[x_margin:x_margin+height, y_margin:y_margin+width,:]


# ----------------------------------------------------------------------------------------------------
# statistical features extraction --------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def discrete_cosine_transform(image:np.array, block_size:int=8):

    idx = 0
    h, w, c = image.shape
    n_h, n_w = h//block_size, w//block_size

    dct_ = np.zeros((n_h*n_w, block_size**2))
    
    for i in range(n_h):
        for j in range(n_w):

            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 0]
            block = np.float32(block)/255.0

            dct_[idx] = cv2.dct(np.float32(block)).flatten()
            idx+=1

    return dct_


def compute_kurtosis(X:np.array):
    
    n, d = X.shape
    kurt = np.apply_along_axis(scipy.stats.kurtosis, 0, X)
    kurt = kurt.reshape((1, d))
    # normalization
    kurt[kurt > 100] = 100
    kurt[kurt < 0] = 0
    kurt = kurt/100
    # kurt = (kurt + np.absolute(kurt.min())) / (kurt.max() - kurt.min())
    
    return kurt
    

# ----------------------------------------------------------------------------------------------------
# configuration file ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def load_config(config_path:str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# ----------------------------------------------------------------------------------------------------
# iso-noise creation ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def is_rgb_image(image):
    return len(image.shape) == 3 and image.shape[-1] == 3

def iso_noise(image, color_shift=0.05, intensity=0.5, random_state=None, **kwargs):
    """
    Apply poisson noise to image to simulate camera sensor noise.
    Args:
        image (numpy.ndarray): Input image, currently, only RGB, uint8 images are supported.
        color_shift (float):
        intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                   yet acceptable level of noise.
        random_state:
        **kwargs:
    Returns:
        numpy.ndarray: Noised image
    """
    if image.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")
    if not is_rgb_image(image):
        raise TypeError("Image must be RGB")
    if color_shift == 0: 
        return (image, None, None)


    one_over_255 = float(1.0 / 255.0)
    image = np.multiply(image, one_over_255, dtype=np.float32)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_utils.poisson(stddev[1] * intensity * 255, size=hls.shape[:2], random_state=random_state)
    color_noise = random_utils.normal(0, color_shift * 360 * intensity, size=hls.shape[:2], random_state=random_state)

    hue = hls[..., 0]
    hue += color_noise
    hue[hue < 0] += 360
    hue[hue > 360] -= 360

    luminance = hls[..., 1]
    luminance += (luminance_noise / 255) * (1.0 - luminance)

    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255
    return (image.astype(np.uint8), luminance_noise, color_noise)

def create_iso_noise(image:np.array):

    config = load_config("/workspace/nem/NEM/configs/configuration.yaml")
    colorshift_range = config['noise_creation']['iso_noise_range']
    colorshift_proba = config['noise_creation']['iso_noise_proba']

    cs = random.choices(colorshift_range, colorshift_proba)[0]

    noise_type = 'iso'
    noise_value = cs

    if cs == 0: return (image, noise_type, noise_value)

    image_noise, _, _ = iso_noise(image, color_shift=0.0027*cs, intensity=0.1, p=1)

    return (image_noise, noise_type, noise_value)


# ----------------------------------------------------------------------------------------------------
# gaussian noise creation ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def create_gaussian_noise(image:np.array):

    config = load_config("/workspace/nem/NEM/configs/configuration.yaml")
    vrc_range = config['noise_creation']['gaussian_noise_range']
    vrc_proba = config['noise_creation']['gaussian_noise_proba']

    v = random.choices(vrc_range, vrc_proba)[0]

    noise_type = 'gaussian'
    noise_value = v

    if v == 0: return (image, noise_type, noise_value)

    transform = A.augmentations.transforms.GaussNoise(var_limit = (v,v), mean = 0, per_channel = True, p=1)
    image_noise = transform(image=image)

    return (image_noise['image'], noise_type, noise_value)


# ----------------------------------------------------------------------------------------------------
# gaussian noise creation ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def create_noise(image:np.array):

    config = load_config("/workspace/nem/NEM/configs/configuration.yaml")
    proba_gaussian = config['noise_creation']['gaussian_iso_balance'][0]
    
    x = np.random.uniform()

    if x <= proba_gaussian : 
        return create_gaussian_noise(image)
    return create_iso_noise(image)
    

# ----------------------------------------------------------------------------------------------------
# noise score mapping  -------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def linear_function(point1:tuple, point2:tuple, x):

    x_range = point2[0] - point1[0]
    y_range = point2[1] - point1[1]

    slope = y_range / x_range
    intercept = point1[1] - point1[0]*slope

    return(x*slope + intercept)


def f(x, values):
    if x == 'None': 
        value = values[0] 
        values.pop(0)
        return value
    else: return x


def create_noise_score(noise_type:str, noise_value:int):

    config = load_config("/workspace/nem/NEM/configs/configuration.yaml")

    if noise_type == 'iso':

        value_list = config['noise_mapping']['iso_value']
        score_list = config['noise_mapping']['iso_score']

    elif noise_type == 'gaussian':

        value_list = config['noise_mapping']['gaussian_value']
        score_list = config['noise_mapping']['gaussian_score']

    points = []
    
    for (idx, v) in enumerate(score_list):

        if v == 'None': 
            
            y1 = score_list[idx-1]
            y2 = score_list[idx+1]
            x1 = value_list[idx][0]
            x2 = value_list[idx][1]
            
            points.insert(len(points), (x1,y1))
            points.insert(len(points), (x2,y2))
    
    values = []
    
    for i in range(len(points)//2):
        
        values.append(linear_function(points[2*i], points[2*i + 1], noise_value))
    
    score_list = list(map(lambda x: f(x, values), score_list))

    # print(f"noise_value : {noise_type} // {type(noise_type)}")
    # print(f"noise_value : {noise_value} // {type(noise_value)}\n")

    # print(f"value_list : {value_list} // ")
    # print(f"score_list : {score_list} // ")

    # print(f"points : {points} // ")
    # print(f"values : {values} // ")

    return int(np.piecewise(noise_value, [j[0]<= noise_value < j[1] for j in value_list], score_list))


def create_noise_score_sigmoid(noise_type:str, noise_value:int):

    # popt_iso = [161, 0.0328, 103]
    popt_iso = [60, 0.1, 100]
    # popt_gaussian = [160, 0.0345, 102]
    popt_gaussian = [140, 0.04, 100]

    if noise_type == 'iso':

        return int(sigmoid(noise_value, popt_iso[0], popt_iso[1], popt_iso[2]))

    elif noise_type == 'gaussian':

        return int(sigmoid(noise_value, popt_gaussian[0], popt_gaussian[1], popt_gaussian[2]))


# ----------------------------------------------------------------------------------------------------
# train-test split -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def train_test_split(csv_path:str, train_split=0.7, test_split=0.2, val_split=0.1):

    df = pd.read_csv(csv_path, index_col=0)
    
    n_images = len(df['img_id'])
    
    n_train = n_images*train_split
    n_test = n_images*test_split
    n_val = n_images*val_split

    shooting_ids = df['shooting_id'].unique()
    
    paths_train = []
    paths_test = []
    paths_val = []

    np.random.shuffle(shooting_ids)

    i=0

    while len(paths_train)<n_train:

        shooting_id = shooting_ids[i]

        image_paths_shooting = df[df['shooting_id'] == shooting_id][['image_path', 'exposure']]
        
        for u in image_paths_shooting.itertuples(): 
            
            paths_train.append((u.image_path, u.exposure))

        i+=1

    while len(paths_test)<n_test:

        shooting_id = shooting_ids[i]

        image_paths_shooting = df[df['shooting_id'] == shooting_id][['image_path', 'exposure']]
        
        for u in image_paths_shooting.itertuples(): 
            
            paths_test.append((u.image_path, u.exposure))

        i+=1

    paths_remaining = df[df['shooting_id'].isin(shooting_ids[i:])][['image_path', 'exposure']]
    
    for u in paths_remaining.itertuples(): 
        
        paths_val.append((u.image_path, u.exposure)) 

    return(paths_train, paths_test, paths_val)

# ----------------------------------------------------------------------------------------------------
# learning rate scheduler  ---------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

class LRScheduler():

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.60):

        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    patience=self.patience,
                    factor=self.factor,
                    min_lr=self.min_lr,
                    verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


# ----------------------------------------------------------------------------------------------------
# list mean function ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def mean(my_list:list):
    return(sum(my_list)/len(my_list))



        


