import random, cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import utils


from pathlib import PurePath
from models import ResNet18_compress1

"""Set up arguments, dataset, model"""
# Set up arguments
def get_args(sample_idx=None):
    def args():
        pass
    args.data_path = './data/dev.npz'
    args.image_size = 256
    args.plane = 'sag'
    args.mri_type = 't2'
    args.num_labels = 2
    args.pretrained = False
    args.load_path = './save/train/resnet_sag_t1-02/best.pth.tar'
    args.return_step = False
    args.seed = 25
    args.m_steps = 50
    args.sample_idx = sample_idx
    
    return args

class TFCCDataset(data.Dataset):
    def __init__(self, data_path, transform, image_size, plane='cor', mri_type='t1'):
        super().__init__()
        data = np.load(data_path, allow_pickle=True)

        self.image_size = image_size
        self.plane = plane
        self.mri_type = mri_type
        self.root = PurePath(data_path).parent
        
        self.files = data['files']
        self.labels = torch.from_numpy(data['labels']).long()
        self.idxs = torch.from_numpy(data['idxs']).long()
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        image_file = self.files[idx] + '.npy'
        image_path = self.root.joinpath(f'{self.plane}_{self.mri_type}', image_file)
        image = np.load(image_path)
        label = self.labels[idx]
        idx = self.idxs[idx]

        image = self.transform(image)
        image = F.interpolate(image.unsqueeze(0), size=[self.image_size, self.image_size], mode='bilinear', align_corners=False).squeeze(0)

        example = (image_path, image, label)

        return example

def normalize(image):
    mean = torch.tensor([0.5] * 8)[:, None, None]
    std = torch.tensor([0.5] * 8)[:, None, None]
    return (image - mean) / std

def interpolate_images(image, baseline, alphas):  
    delta = image - baseline
    interpolated_images = baseline + (alphas[:, None, None, None] * delta)
    return interpolated_images

def compute_gradients(images, target_class_idx, model):
    # Forward pass
    images = nn.Parameter(images, requires_grad=True)
    target_class_idxs = torch.full([images.size(0)], target_class_idx.item(), dtype=torch.int64)
    log_p = model(images)
    loss = F.nll_loss(log_p, target_class_idxs)
    
    # Backward pass
    loss.backward()

    return images.grad

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / 2
    integrated_gradients = grads.mean(dim=0)
    return integrated_gradients

def integrated_gradients(baseline, image, model, target_class_idx, m_steps=50, batch_size=32):
    """Compute integrated gradients"""
    # Generate alphas
    alphas = torch.linspace(start=0, end=1, steps=m_steps+1)
    
    # Initialize batch gradients list outside loop to collect gradients
    gradient_batches = []
    
    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps
    for alpha in range(0, len(alphas), batch_size):
        from_ = alpha
        to = min(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        
        # Generate interpolated inputs between baseline and input
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           image=image,
                                                           alphas=alpha_batch)
        
        # Compute gradients between model outputs and interpolated inputs
        gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                           target_class_idx=target_class_idx,
                                           model=model)
        
        # Append each batch gradient to batches gradient list
        gradient_batches.append(gradient_batch)
    
    # concatenate path gradients together row-wise into single tensor
    total_gradients = torch.cat(gradient_batches, dim=0)
    
    # Integral approximation through averaging gradients
    avg_gradients = integral_approximation(gradients=total_gradients)
    
    # Scale integrated gradients with respect to input
    integrated_gradients = (image - baseline) * avg_gradients
    
    return integrated_gradients

def save_img_attributions(baseline, image, target_class_idx, plane, model, m_steps=50, cmap=None, overlay_alpha=0.4):
    attributions = integrated_gradients(baseline=baseline,
                                        image=image,
                                        model=model,
                                        target_class_idx=target_class_idx,
                                        m_steps=m_steps)
    # Mask the attributions less than 0
    attributions_smooth = ndimage.gaussian_filter(attributions, sigma=(1.5, 1.5, 0), order=0)
    attribution_mask = np.where(attributions_smooth > 0, attributions_smooth, 0)
    
    for channel in range(image.shape[0]):
        fig = plt.figure()
        plt.imshow(attribution_mask[channel, :, :], cmap=cmap)
        plt.imshow(image[channel, :, :], alpha=overlay_alpha, cmap='gray')
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        plt.savefig(f'./save/{plane}_{channel}_ig.png')
    np.savez("./save/ig.npz", attribution=attributions, image=image.cpu().detach().numpy())


def main():
    args = get_args()

    # Load dataset
    transform_composed = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5]*8,
                                                                std=[0.5]*8)])
    dataset = TFCCDataset(args.data_path, transform_composed, args.image_size,
                        plane=args.plane, mri_type=args.mri_type)

    # Load model
    device, args.gpu_ids = utils.get_available_devices()
    model = ResNet18_compress1(num_labels=args.num_labels, pretrained=args.pretrained)
    model = nn.DataParallel(model, args.gpu_ids)
    model = utils.load_model(model, args.load_path, args.gpu_ids, return_step=args.return_step)
    model.eval()

    # sample index
    random.seed(args.seed)
    sample_idx = random.choice(range(len(dataset)))
    image_path, image, label = dataset[sample_idx]

    # baseline
    baseline = torch.zeros(image.shape).to(device)

    save_img_attributions(baseline=normalize(baseline), image=image, target_class_idx=label,
                            plane=args.plane, model=model, m_steps=300, 
                            cmap=plt.cm.inferno, overlay_alpha=0.4)

if __name__ == '__main__':
    main()