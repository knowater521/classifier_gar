import re
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import moxing as mox
import torch as t
mox.file.shift('os', 'mox')
class DataClassify(Dataset):
    def __init__(self, root, transforms=None, mode=None):
        
        self.imgs = [x.path for x in mox.file.scan_dir(root) if
            x.name.endswith(".jpg")]
        self.labels = [y.path for y in mox.file.scan_dir(root) if
            y.name.endswith(".txt")]
        self.transforms = transforms
        
    def __getitem__(self, index):
        
        img_path = self.imgs[index]
        '''
        with open(self.labels[index]) as file_pro:
            contents = file_pro.read()
            print(contents)
        '''
        label = int(re.sub('\D', '', open(self.labels[index]).read()[-4:]))
        #label = int(re.sub('\D', '', mox.file.read(mox.file.File(self.labels[index]))[-4:]))
        #label = int(re.sub('\D','',mox.file.read(self.labels[index])[-4:]))
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    
    def __len__(self):
        return len(self.imgs)


def ListToTensor(lister):
    nparr = np.asarray(lister)
    tens = t.from_numpy(nparr)
    return tens