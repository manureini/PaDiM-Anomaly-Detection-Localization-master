import random
from random import sample
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import time
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
from skimage.io import imread, imsave
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
from torchvision import transforms as T
from PIL import Image
import datasets.mvtec as mvtec

class PaDiMDetector(object):
    """description of class"""

    def __init__(self, arch='wide_resnet50_2', random_seed=1024):
        self.arch = arch
        self.random_seed = random_seed

        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # load model
        if self.arch == 'resnet18':
            self.model = resnet18(pretrained=True, progress=True)
            t_d = 448
            d = 100
        elif self.arch == 'wide_resnet50_2':
            self.model = wide_resnet50_2(pretrained=True, progress=True)
            t_d = 1792
            d = 550

        self.model.to(self.device)
        self.model.eval()

        self.idx = torch.tensor(sample(range(0, t_d), d))

        self.transform_x = T.Compose([T.Resize(256, Image.ANTIALIAS),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])


    def train(self, data_path, class_name):
        # set model's intermediate outputs
        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        train_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])   
        
        for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            # clear hook outputs
            outputs = []

        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, train_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()

        I = np.identity(C)

        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :,
            # i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I

        # save learned distribution
        self.train_outputs = [mean, cov]

    def save_model(self, file_path='model.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(self.train_outputs, f)
        with open(file_path + '.values', 'wb') as f:
            pickle.dump([self.min_score, self.max_score, self.threshold], f)

    def load_model(self, file_path='model.pkl'):
        with open(file_path, 'rb') as f:
            self.train_outputs = pickle.load(f)
        with open(file_path + '.values', 'rb') as f:
            self.min_score, self.max_score, self.threshold = pickle.load(f)

    def check(self, file_name):
        outputs = []
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        x = Image.open(file_name).convert('RGB')
        x = self.transform_x(x)
        x = x.unsqueeze(0)
        #x.shape == ([1, 3, 224, 224])

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        with torch.no_grad():
            _ = self.model(x.to(self.device))

        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        #1, 256, 56, 56
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []

        for i in range(H * W):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()
        
        score_map = gaussian_filter(score_map, sigma=4)
        
        # Normalization
        scores = (score_map - self.min_score) / (self.max_score - self.min_score)

        img_scores = scores.max()
        
        #heat_map = scores * 255
        #imsave("out.png", heat_map)

        return img_scores > threshold

    def evaluate(self, data_path, class_name, save_path):
        test_dataset = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        
        gt_list = []
        gt_mask_list = []
        test_imgs = []    

        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))

            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())

            # clear hook outputs
            outputs = []

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []

        for i in range(H * W):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        self.max_score = score_map.max()
        self.min_score = score_map.min()
        scores = (score_map - self.min_score) / (self.max_score - self.min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_img_rocauc = ax[0]
        fig_pixel_rocauc = ax[1]
           
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        self.threshold = thresholds[np.argmax(f1)]
        print('Threshold: ' + str(self.threshold))

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = save_path + '/' + f'pictures_{self.arch}'
        os.makedirs(save_dir, exist_ok=True)
        self.plot_fig(test_imgs, scores, gt_mask_list, self.threshold, save_dir, class_name)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'roc_curve.png'), dpi=100)


    def embedding_concat(self, x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z
    
    def denormalization(self, x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)    
        return x

    def plot_fig(self, test_img, scores, gts, threshold, save_dir, class_name):
        num = len(scores)
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.
        for i in range(num):
            img = test_img[i]
            img = self.denormalization(img)
            gt = gts[i].transpose(1, 2, 0).squeeze()
            heat_map = scores[i] * 255
            mask = scores[i]
            mask[mask > threshold] = 1
            mask[mask <= threshold] = 0
            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
            fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
            fig_img.subplots_adjust(right=0.9)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')
            ax_img[1].imshow(gt, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[2].imshow(img, cmap='gray', interpolation='none')
            ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax_img[2].title.set_text('Predicted heat map')
            ax_img[3].imshow(mask, cmap='gray')
            ax_img[3].title.set_text('Predicted mask')
            ax_img[4].imshow(vis_img)
            ax_img[4].title.set_text('Segmentation result')
            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)
    
            fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
            plt.close()
    

if __name__ == '__main__':

    datasetpath = r"D:\Owncloud\HSO\INFM3\images"
    class_name = "orbiter_v2"

    detector = PaDiMDetector()
    detector.train(datasetpath, class_name)
    detector.evaluate(datasetpath, class_name, "results")
    detector.save_model()
    detector.load_model()
    
    start = time.time()
    result = detector.check(r"D:\Owncloud\HSO\INFM3\images\orbiter_v2\test\metal\P01297_niO_000383_Ab_3.png")
    end = time.time()
    print(str(end - start) + "s")
    print(result)

    result = detector.check(r"D:\Owncloud\HSO\INFM3\images\orbiter_v2\test\good\P01297_niO_000271_Ab_0.png")
    print(result)

    