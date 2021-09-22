#%%
from transformers import ViTFeatureExtractor, ViTModel
import torchvision
from torch.nn import Module
from torch import nn
import torch as th
import numpy as np
from utils import overlay_predictions_on_images

class RoadDetector(Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.patch_size = 16
        self.n_classes = n_classes
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.decoder = nn.Linear(768, self.patch_size**2*self.n_classes)

    def forward(self, x, device):
        inputs = self.feature_extractor(images=x, return_tensors="pt")
        bsize, _, height, width = inputs['pixel_values'].shape
        inputs = {k:inputs[k].to(device) for k in inputs.keys()}
        outputs = self.encoder(**inputs)
        patch_encodings = outputs.last_hidden_state[:, 1:] 
        decoded_patchs = self.decoder(patch_encodings)
        decoded_patchs = decoded_patchs.view(bsize, height, width, self.n_classes)
        return th.softmax(decoded_patchs, dim=-1)

    def visualize(self, frame, device, middles=True):
        if type(frame) != type(np.array([1])):
            frame = np.array(frame)
        img = th.tensor(frame.transpose(2, 0, 1))
        img = torchvision.transforms.Resize((224, 224))(img).unsqueeze(0)
        img = img.transpose(1, 3).transpose(1, 2)
        with th.no_grad():
            preds = self([frame], device)
        overlaid = overlay_predictions_on_images(img, preds, alpha=0.7)
        overlaid = overlaid[0].astype(np.uint8)

        if middles:
            pred = preds[0, :, :, 0].detach().cpu().numpy()
            idxs = np.arange(len(pred))
            idxs = np.tile(idxs, (len(pred[0]), 1))
            line = (np.sum(pred*idxs, axis=1)/np.sum(pred, axis=1)).astype(np.uint8)
            for i, j in enumerate(line):
                overlaid[i, j] = (255, 0, 0)
            return overlaid, preds, line

        return overlaid, preds

# %%
if __name__ == '__main__':
    import requests
    from PIL import Image

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print("using", device)

    model = RoadDetector().to(device)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    
    # %%
    import time
    t = time.time()
    with th.no_grad():
        a = model([image]*1, device)
    time.time()-t

    # %%
    import matplotlib.pyplot as plt
    plt.imshow(a[0].detach().cpu().numpy(), vmin=0, vmax=1)
    a[0]
