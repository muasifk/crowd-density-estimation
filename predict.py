
import os
from glob import glob
import numpy as np
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt
import urllib.request


''' Load pretrained model '''
from src.models.CSRNet import CSRNet
model       = CSRNet()
model_name  = model.__class__.__name__
# PATH        = f'{os.getcwd()}/checkpoints/{model_name}.pth'  ##  Checkpoint size is larger for github
# state_dict  = torch.load(PATH, map_location=torch.device('cpu'))
PATH        = 'https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth' # Lets import checkpoints from a URL
state_dict  = torch.hub.load_state_dict_from_url(PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
print('\n Model loaded successfully.. \n')
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



''' Predict on test images from original dataset  '''
i = 20 # change to see prediction on new images
# root_dir        = 'path/to/dataset/dir/' 
# test_img_paths = glob(root_dir + '/test_data/images' + '/*.jpg')
# img = cv2.cvtColor(cv2.imread(test_img_paths[i]), cv2.COLOR_BGR2RGB)

''' Read image from Internet '''
# url          = 'https://jooinn.com/images/human-crowd-2.jpg'
url          = 'https://upload.wikimedia.org/wikipedia/commons/8/88/The_million_march_man.jpg'
url_response = urllib.request.urlopen(url)
img_array    = np.array(bytearray(url_response.read()), dtype=np.uint8)
img          = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## Predict
img = img.astype(np.float32)/255 # normalize image
img = torch.from_numpy(img).permute(2,0,1) # reshape to [c,w,h]
et  = model(img.unsqueeze(0))
out = et.squeeze().detach().numpy()
fig, (ax0, ax1) = plt.subplots(1,2, figsize=(6,3))
ax0.imshow(img.permute(1,2,0)) # reshape back to [w,h,c]
ax1.imshow(out, cmap='jet')
ax0.set_title('Image')
ax1.set_title(f'Count: {et.sum():.0f}')
ax0.axis("off")
ax1.axis("off")
plt.show()
plt.savefig('./predictions/new.jpg', dpi=150 )



