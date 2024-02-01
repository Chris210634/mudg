from source.samplers import BaseDataset
import os

def replace_underscores(s):
        new_s = ''
        for si in s:
            if si == '_':
                new_s += ' '
            else:
                new_s += si
        return new_s
    
class DomainNetDataset(BaseDataset):
    ''' Only purpose of this class is to access classnames.'''
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    folders = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 
               'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 
               'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 
               'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 
               'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 
               'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 
               'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 
               'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 
               'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 
               'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 
               'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 
               'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 
               'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 
               'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 
               'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 
               'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 
               'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 
               'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 
               'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 
               'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 
               'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 
               'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 
               'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip', 
               'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 
               'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 
               'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 
               'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 
               'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 
               'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 
               'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 
               'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 
               'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 
               'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 
               'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 
               'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 
               'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 
               'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 
               'yoga', 'zebra', 'zigzag']
    classnames = [replace_underscores(s) for s in folders]

    def __init__(self, config):
        self.transform = None # need to set later
        self.classnames = DomainNetDataset.classnames
        
    def __len__(self):
        raise NotImplemented
        
    def __getitem__(self, index):
        raise NotImplemented
        
def _initialize_domainnet_dataset(root):
    for folder in DomainNetDataset.folders:
        assert folder in os.listdir(root)
    assert len(DomainNetDataset.folders) == len(DomainNetDataset.classnames)
    imgs = []
    for i, folder in enumerate(DomainNetDataset.folders):
        parent_folder = os.path.join(root, folder)
        for img_file in os.listdir(parent_folder):
            # os.path.join(parent_folder, img_file) is path to image file
            # i is label
            imgs.append((os.path.join(parent_folder, img_file), i))
    return imgs

class DomainNetClipartDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = DomainNetDataset.classnames
        self.domain = 'clipart'
        assert self.domain in DomainNetDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'domain_net'))
        self.root = os.path.join(cfg.ROOT, 'domain_net', self.domain)
        self.imgs = _initialize_domainnet_dataset(self.root)
    
class DomainNetInfographDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = DomainNetDataset.classnames
        self.domain = 'infograph'
        assert self.domain in DomainNetDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'domain_net'))
        self.root = os.path.join(cfg.ROOT, 'domain_net', self.domain)
        self.imgs = _initialize_domainnet_dataset(self.root)
        
class DomainNetPaintingDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = DomainNetDataset.classnames
        self.domain = 'painting'
        assert self.domain in DomainNetDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'domain_net'))
        self.root = os.path.join(cfg.ROOT, 'domain_net', self.domain)
        self.imgs = _initialize_domainnet_dataset(self.root)
        
class DomainNetQuickdrawDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = DomainNetDataset.classnames
        self.domain = 'quickdraw'
        assert self.domain in DomainNetDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'domain_net'))
        self.root = os.path.join(cfg.ROOT, 'domain_net', self.domain)
        self.imgs = _initialize_domainnet_dataset(self.root)
      
class DomainNetRealDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = DomainNetDataset.classnames
        self.domain = 'real'
        assert self.domain in DomainNetDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'domain_net'))
        self.root = os.path.join(cfg.ROOT, 'domain_net', self.domain)
        self.imgs = _initialize_domainnet_dataset(self.root)
        
class DomainNetSketchDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = DomainNetDataset.classnames
        self.domain = 'sketch'
        assert self.domain in DomainNetDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'domain_net'))
        self.root = os.path.join(cfg.ROOT, 'domain_net', self.domain)
        self.imgs = _initialize_domainnet_dataset(self.root)

########################################################################################

class OfficeHomeDataset(BaseDataset):
    ''' 
    Only purpose of this class is to access classnames.
    Remember to rename "Real World" folder to RealWorld !!!
    '''
    domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    folders = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
               'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
               'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 
               'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 
               'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 
               'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 
               'Trash_Can', 'Webcam']
    classnames = ['alarm clock', 'backpack', 'batteries', 'bed', 'bike', 'bottle', 'bucket', 'calculator', 'calendar', 'candles', 
               'chair', 'clipboards', 'computer', 'couch', 'curtains', 'desk lamp', 'drill', 'eraser', 'exit sign', 'fan', 
               'file cabinet', 'flipflops', 'flowers', 'folder', 'fork', 'glasses', 'hammer', 'helmet', 'kettle', 'keyboard', 
               'knives', 'lamp shade', 'laptop', 'marker', 'monitor', 'mop', 'mouse', 'mug', 'notebook', 'oven', 'pan', 'paper clip', 
               'pen', 'pencil', 'postit notes', 'printer', 'push pin', 'radio', 'refrigerator', 'ruler', 'scissors', 'screwdriver', 
               'shelf', 'sink', 'sneakers', 'soda', 'speaker', 'spoon', 'TV', 'table', 'telephone', 'toothbrush', 'toys', 
               'trash can', 'webcam']
#     classnames = [replace_underscores(s) for s in folders]

    def __init__(self, config):
        self.transform = None # need to set later
        self.classnames = OfficeHomeDataset.classnames
        
    def __len__(self):
        raise NotImplemented
        
    def __getitem__(self, index):
        raise NotImplemented
    
def _initialize_officehome_dataset(root):
    for folder in OfficeHomeDataset.folders:
        assert folder in os.listdir(root)
    assert len(OfficeHomeDataset.folders) == len(OfficeHomeDataset.classnames)
    imgs = []
    for i, folder in enumerate(OfficeHomeDataset.folders):
        parent_folder = os.path.join(root, folder)
        for img_file in os.listdir(parent_folder):
            # os.path.join(parent_folder, img_file) is path to image file
            # i is label
            imgs.append((os.path.join(parent_folder, img_file), i))
    return imgs

class OfficeHomeArtDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = OfficeHomeDataset.classnames
        self.domain = 'Art'
        assert self.domain in OfficeHomeDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'office_home'))
        self.root = os.path.join(cfg.ROOT, 'office_home', self.domain)
        self.imgs = _initialize_officehome_dataset(self.root)
    
class OfficeHomeClipartDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = OfficeHomeDataset.classnames
        self.domain = 'Clipart'
        assert self.domain in OfficeHomeDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'office_home'))
        self.root = os.path.join(cfg.ROOT, 'office_home', self.domain)
        self.imgs = _initialize_officehome_dataset(self.root)
        
class OfficeHomeProductDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = OfficeHomeDataset.classnames
        self.domain = 'Product'
        assert self.domain in OfficeHomeDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'office_home'))
        self.root = os.path.join(cfg.ROOT, 'office_home', self.domain)
        self.imgs = _initialize_officehome_dataset(self.root)
        
class OfficeHomeRealDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = OfficeHomeDataset.classnames
        self.domain = 'RealWorld'
        assert self.domain in OfficeHomeDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'office_home'))
        self.root = os.path.join(cfg.ROOT, 'office_home', self.domain)
        self.imgs = _initialize_officehome_dataset(self.root)
    
########################################################################################

class PACSDataset(BaseDataset):
    ''' 
    Only purpose of this class is to access classnames.
    Remember to rename "Real World" folder to RealWorld !!!
    '''
    domains = ['sketch', 'cartoon', 'art_painting', 'photo']
    folders = ['horse', 'person', 'house', 'dog', 'giraffe', 'guitar', 'elephant']
    classnames = ['horse', 'person', 'house', 'dog', 'giraffe', 'guitar', 'elephant']
    
    def __init__(self, config):
        self.transform = None # need to set later
        self.classnames = PACSDataset.classnames
        
    def __len__(self):
        raise NotImplemented
        
    def __getitem__(self, index):
        raise NotImplemented
    
def _initialize_pacs_dataset(root):
    for folder in PACSDataset.folders:
        assert folder in os.listdir(root)
    assert len(PACSDataset.folders) == len(PACSDataset.classnames)
    imgs = []
    for i, folder in enumerate(PACSDataset.folders):
        parent_folder = os.path.join(root, folder)
        for img_file in os.listdir(parent_folder):
            # os.path.join(parent_folder, img_file) is path to image file
            # i is label
            imgs.append((os.path.join(parent_folder, img_file), i))
    return imgs

class PACSArtDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = PACSDataset.classnames
        self.domain = 'art_painting'
        assert self.domain in PACSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'PACS'))
        self.root = os.path.join(cfg.ROOT, 'PACS', self.domain)
        self.imgs = _initialize_pacs_dataset(self.root)
    
class PACSCartoonDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = PACSDataset.classnames
        self.domain = 'cartoon'
        assert self.domain in PACSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'PACS'))
        self.root = os.path.join(cfg.ROOT, 'PACS', self.domain)
        self.imgs = _initialize_pacs_dataset(self.root)
        
class PACSPhotoDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = PACSDataset.classnames
        self.domain = 'photo'
        assert self.domain in PACSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'PACS'))
        self.root = os.path.join(cfg.ROOT, 'PACS', self.domain)
        self.imgs = _initialize_pacs_dataset(self.root)
        
class PACSSketchDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = PACSDataset.classnames
        self.domain = 'sketch'
        assert self.domain in PACSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'PACS'))
        self.root = os.path.join(cfg.ROOT, 'PACS', self.domain)
        self.imgs = _initialize_pacs_dataset(self.root)
        
########################################################################################

class VLCSDataset(BaseDataset):
    ''' 
    Only purpose of this class is to access classnames.
    Remember to rename "Real World" folder to RealWorld !!!
    '''
    domains = ['Caltech101', 'VOC2007', 'LabelMe', 'SUN09']
    folders = ['car', 'chair', 'person', 'dog', 'bird']
    classnames = ['car', 'chair', 'person', 'dog', 'bird']
    
    def __init__(self, config):
        self.transform = None # need to set later
        self.classnames = VLCSDataset.classnames
        
    def __len__(self):
        raise NotImplemented
        
    def __getitem__(self, index):
        raise NotImplemented
    
def _initialize_vlcs_dataset(root):
    for folder in VLCSDataset.folders:
        assert folder in os.listdir(root)
    assert len(VLCSDataset.folders) == len(VLCSDataset.classnames)
    imgs = []
    for i, folder in enumerate(VLCSDataset.folders):
        parent_folder = os.path.join(root, folder)
        for img_file in os.listdir(parent_folder):
            # os.path.join(parent_folder, img_file) is path to image file
            # i is label
            imgs.append((os.path.join(parent_folder, img_file), i))
    return imgs

class VLCSCaltechDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = VLCSDataset.classnames
        self.domain = 'Caltech101'
        assert self.domain in VLCSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'VLCS'))
        self.root = os.path.join(cfg.ROOT, 'VLCS', self.domain)
        self.imgs = _initialize_vlcs_dataset(self.root)
    
class VLCSLabelmeDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = VLCSDataset.classnames
        self.domain = 'LabelMe'
        assert self.domain in VLCSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'VLCS'))
        self.root = os.path.join(cfg.ROOT, 'VLCS', self.domain)
        self.imgs = _initialize_vlcs_dataset(self.root)
        
class VLCSSunDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = VLCSDataset.classnames
        self.domain = 'SUN09'
        assert self.domain in VLCSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'VLCS'))
        self.root = os.path.join(cfg.ROOT, 'VLCS', self.domain)
        self.imgs = _initialize_vlcs_dataset(self.root)
        
class VLCSVocDataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = VLCSDataset.classnames
        self.domain = 'VOC2007'
        assert self.domain in VLCSDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'VLCS'))
        self.root = os.path.join(cfg.ROOT, 'VLCS', self.domain)
        self.imgs = _initialize_vlcs_dataset(self.root)
        
########################################################################################

class TerraIncDataset(BaseDataset):
    ''' 
    Only purpose of this class is to access classnames.
    Remember to rename "Real World" folder to RealWorld !!!
    '''
    domains = ['location_43', 'location_46', 'location_38', 'location_100']
    folders = ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon', 'squirrel']
    classnames = ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon', 'squirrel']
    
    def __init__(self, config):
        self.transform = None # need to set later
        self.classnames = TerraIncDataset.classnames
        
    def __len__(self):
        raise NotImplemented
        
    def __getitem__(self, index):
        raise NotImplemented
    
def _initialize_terrainc_dataset(root):
    for folder in TerraIncDataset.folders:
        assert folder in os.listdir(root)
    assert len(TerraIncDataset.folders) == len(TerraIncDataset.classnames)
    imgs = []
    for i, folder in enumerate(TerraIncDataset.folders):
        parent_folder = os.path.join(root, folder)
        for img_file in os.listdir(parent_folder):
            # os.path.join(parent_folder, img_file) is path to image file
            # i is label
            imgs.append((os.path.join(parent_folder, img_file), i))
    return imgs

class TerraInc100Dataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = TerraIncDataset.classnames
        self.domain = 'location_100'
        assert self.domain in TerraIncDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'terra_incognita'))
        self.root = os.path.join(cfg.ROOT, 'terra_incognita', self.domain)
        self.imgs = _initialize_terrainc_dataset(self.root)
    
class TerraInc38Dataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = TerraIncDataset.classnames
        self.domain = 'location_38'
        assert self.domain in TerraIncDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'terra_incognita'))
        self.root = os.path.join(cfg.ROOT, 'terra_incognita', self.domain)
        self.imgs = _initialize_terrainc_dataset(self.root)
        
class TerraInc43Dataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = TerraIncDataset.classnames
        self.domain = 'location_43'
        assert self.domain in TerraIncDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'terra_incognita'))
        self.root = os.path.join(cfg.ROOT, 'terra_incognita', self.domain)
        self.imgs = _initialize_terrainc_dataset(self.root)
        
class TerraInc46Dataset(BaseDataset):
    def __init__(self, cfg):
        ''' 
        test only. Need to set self.transform after calling constructor.
        Expect: cfg.ROOT = args.data_dir
            This is the dir containing the domain directories
        '''
        self.metadata = None
        self.transform = None # need to set later!!!
        self.classnames = TerraIncDataset.classnames
        self.domain = 'location_46'
        assert self.domain in TerraIncDataset.domains
        assert self.domain in os.listdir(os.path.join(cfg.ROOT, 'terra_incognita'))
        self.root = os.path.join(cfg.ROOT, 'terra_incognita', self.domain)
        self.imgs = _initialize_terrainc_dataset(self.root)
        
########################################################################################