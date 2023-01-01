import glob

from os.path import join
from torch.utils.data import Dataset
from PIL import Image

class StyleDataset(Dataset):    
    def __init__(self, 
                 transforms,
                 path="./data/"):
        """
        ModelProductDataset holds dataset for images crawled from fashion websites. 

        :param transforms:  transforms to apply to the target image 
        :param path:        path to folder containing directory 'split' with subdirectories model and product
        :param split:       dataset split to use, training or validation
        """
        
        self.path = path
        self.transforms = transforms
        self.data = self.initialize_dataset()

    def __len__(self):
        """
        returns length of dataset
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """ 
        gets image at index in the dataset
        """
        path = self.data[index]
    
        img = Image.open(path).convert('RGB') 

        img = self.transforms(img)

        return img
    
    def initialize_dataset(self):
        """
        helper function to aggregate image files
        """
        files = sorted(glob.glob(self.path + "*"))
        return files