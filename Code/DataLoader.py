class MammogramDataset_Custom(Dataset):   
    def __init__(self, csv_file, root_dir, image_column, num_channel=1, transform = None, 
                 transform_type = 'Custom', transform_prob=0.5):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column: column name from csv file where we take the file path
        """
        #self.data_frame = pickle.load(open(os.path.join(root_dir,data_file),"rb"))
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_column = image_column
        self.num_channel = num_channel
        self.transform_prob = transform_prob
        self.transform_type = transform_type

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_frame.loc[idx, self.image_column]))
        image = pydicom.dcmread(img_name).pixel_array
        h,w = image.shape
        pad_row = 7500-h
        pad_col = 5500-w
        if sum(image[:,-1]) == 0:
            image = np.pad(image,((0,pad_row),(0,pad_col)),mode='constant',constant_values=0)
        else:
            image = np.pad(image,((0,pad_row),(pad_col,0)),mode='constant',constant_values=0)
        image = np.float32(image/np.iinfo(image.dtype).max)

        image = (image - 0.3328) / 0.7497
        if self.num_channel > 1:
            image=np.repeat(image[None,...],self.num_channel,axis=0)
        
        image_class = self.data_frame.loc[idx, 'class']

        if self.transform:
            image = self.transform(image)
        elif self.transform_type == 'Custom':
            p1 = random.uniform(0, 1)
            p2 = random.uniform(0, 1)
            if p1 <= self.transform_prob:
                if p2 <= self.transform_prob:
                    image = np.flip(image,0).copy()
                else:
                    image = np.flip(image,1).copy()
            
        
        sample = {'x': image[None,:], 'y': image_class}
        return sample


def GetDataLoader_TL(train_csv, validation_csv, test_csv, 
                     root_dir, image_column, num_channel, 
                     transform_type, transform_prob, 
               train_transform, validation_transform, 
               batch_size, shuffle, num_workers): 

    train_data = MammogramDataset_Custom(csv_file = train_csv, 
                              root_dir = root_image,
                              image_column = image_column,
                              num_channel = num_channel, 
                               transform=train_transform, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    val_data = MammogramDataset_Custom(csv_file = validation_csv, 
                            root_dir = root_image,
                            image_column = image_column,
                            transform = validation_transform,
                                 num_channel = num_channel, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    test_data = MammogramDataset_Custom(csv_file = test_csv, 
                            root_dir = root_image,
                            image_column = image_column,
                            transform = validation_transform,
                            num_channel = num_channel, 
                               transform_type = transform_type, 
                                   transform_prob = transform_prob)
    
    image_datasets = {'train': train_data, 'val': val_data, 'test': test_data}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True, 
                                              num_workers=NUM_WORKERS) 
                    for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    return dataloaders, dataset_sizes