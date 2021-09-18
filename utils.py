#%%
import torch as th
import numpy as np

def load_model(model, path):
    try:
        model.load_state_dict(th.load(path))
        print(f"loaded model ({type(model).__name__}) from {path}")
    except:
        print(f"could not load model ({type(model).__name__}) from {path}")

def save_model(model, path):
    th.save(model.state_dict(), path)

def collate(samples):
    imgs, labels = [], []
    for img, label in samples:
        imgs.append(img)
        labels.append(label)
    labels = th.cat(labels, dim=0)
    return imgs, labels

def overlay_predictions_on_images(imgs, preds, alpha=0.8):
    base = [np.array(img) for img in imgs]
    base = [np.expand_dims(img, axis=0) for img in imgs]
    base = np.concatenate(base, axis=0)
    preds = th.argmax(preds, dim=-1, keepdim=True).detach().cpu().numpy()
    viz = base * (preds==2) + \
        base*alpha*(preds!=2) + \
        np.array([[[0, 255, 0]]])*(preds==0)*(1-alpha) + \
        np.array([[[0, 0, 255]]])*(preds==1)*(1-alpha)
    return viz
# %%
