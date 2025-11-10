import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

##import user lib
from base.data import load_data
from base.utils import update_config , ClipLoss, instantiate_from_config, get_device
import torch.nn.functional as F
# device = get_device('auto')

def load_model(config,train_loader,test_loader):
    model = {}
    for k,v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)
    pl_model = PLModel(model,config,train_loader,test_loader)
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader, model_type='RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)

        self.lambda_img2img = config['lambda_img2img']
        self.lambda_img2eeg = config['lambda_img2eeg']
        self.lambda_mmd = config['lambda_mmd']
        self.lambda_map = config['lambda_map']

        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['z_dim']

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3

        self.mAP_total = 0
        self.match_similarities = []

    ### MMD Loss
    def compute_mmd(self, x, y, kernel='rbf', sigma=1.0):
        """Compute Maximum Mean Discrepancy (MMD) with RBF kernel"""
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K_xx = torch.exp(- (rx.t() + rx - 2 * xx) / (2 * sigma ** 2))
        K_yy = torch.exp(- (ry.t() + ry - 2 * yy) / (2 * sigma ** 2))
        K_xy = torch.exp(- (rx.t() + ry - 2 * xy) / (2 * sigma ** 2))

        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd
    def compute_map_loss(self, similarity, labels):
        batch_size = similarity.shape[0]
        ap_list = []

        for i in range(batch_size):
            true_label = labels[i]
            sims = similarity[i]
            sorted_indices = torch.argsort(-sims)
            sorted_labels = labels[sorted_indices]
            positive_mask = (sorted_labels == true_label).float()
            num_positives = positive_mask.sum()
            if num_positives > 0:
                precision_at_k = torch.cumsum(positive_mask, dim=0) / (torch.arange(1, batch_size + 1).float().to(sims.device))
                ap = (precision_at_k * positive_mask).sum() / num_positives
                ap_list.append(ap)
        if len(ap_list) > 0:
            map_loss = 1 - torch.stack(ap_list).mean()  # maximize mAP
        else:
            map_loss = torch.tensor(0.0, device=similarity.device)

        return map_loss

    def forward(self, batch, sample_posterior=False):
        idx = batch['idx'].cpu().detach().numpy()
        eeg = batch['eeg']
        img = batch['img']
        batch_size = batch['idx'].shape[0]

        img_z = batch['img_features']

        # use img_encoder
        img_z_proj = self.shrink_adapter(img_z)
        img_z_proj = img_z_proj / img_z_proj.norm(dim=-1, keepdim=True)

        eeg_z = self.brain(eeg)
        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)
        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z_proj, logit_scale)
        clip_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        eeg_cos_similarity_matrix = torch.mm(eeg_z, eeg_z.transpose(0, 1))  # [batch_size, batch_size]
        img_cos_similarity_matrix = torch.mm(img_z_proj, img_z_proj.transpose(0, 1))  # [batch_size, batch_size]
        eeg_img_cos_similarity = F.cosine_similarity(eeg_cos_similarity_matrix, img_cos_similarity_matrix)

        img_original_cos_similarity_matrix = torch.mm(
            img_z,
            img_z.transpose(0, 1)
        )  # [batch_size, batch_size]

        # compution cosine similarity between img_z and img_z_proj
        img_img_original_cos_similarity = F.cosine_similarity(img_cos_similarity_matrix, img_original_cos_similarity_matrix)

        mmd_loss = self.compute_mmd(eeg_z, img_z_proj, sigma=1.0)

        # compute mAP loss
        # compute RSM
        similarity = (eeg_z @ img_z_proj.T)
        labels = batch['label']
        map_loss = self.compute_map_loss(similarity, labels)


        # LOSS SCE add semantic distribution consistency loss
        total_loss = clip_loss - self.lambda_img2eeg*eeg_img_cos_similarity.mean() - self.lambda_img2img*img_img_original_cos_similarity.mean() + self.lambda_mmd * mmd_loss + self.lambda_map * map_loss
        self.log('clip_loss', clip_loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('mmd_loss', mmd_loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('map_loss', map_loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('eeg_img_cos_sim_loss', eeg_img_cos_similarity.mean(), on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('img_img_original_cos_sim_loss', img_img_original_cos_similarity.mean(), on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        
        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]

            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label

            loss = total_loss
        else:
            loss = total_loss
        return eeg_z, img_z_proj, loss

    
    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch,sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            counter = Counter(self.match_label)
            count_dict = dict(counter)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True,logger=True, sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label

        return loss


    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
    
        eeg_z, img_z, loss= self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss
    
    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self,batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label =  batch['label']
        self.all_true_labels.extend(label.cpu().numpy())

        #compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())


        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap
        
        return loss
        
    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return  {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),'test_top5_acc':top_k_accuracy.item(),'mAP':self.mAP,'similarity':self.match_similarities}
        

    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 

        return [optimizer], [scheduler]

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="ats.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-01',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=150,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help="vision_backbone",
    )
    parser.add_argument(
        "--adaptor_backbone",
        type=str,
        help="adaptor_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to use for training (e.g., 'cpu', 'cuda:0', 'auto')",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1024,
        help="Training batch size",
    )
    parser.add_argument(
        "--lambda_img2img",
        type=float,
        default=0.0,
        help="Weight for image-image cosine similarity loss",
    )
    parser.add_argument(
        "--lambda_img2eeg",
        type=float,
        default=0.0,
        help="Weight for EEG-image cosine similarity loss",
    )
    parser.add_argument(
        "--lambda_mmd",
        type=float,
        default=0.0,
        help="Weight for mmd loss",
    )
    parser.add_argument(
        "--lambda_map",
        type=float,
        default=0.0,
        help="Weight for map loss",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]
    config['data']['train_batch_size'] = opt.train_batch_size

    if opt.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(opt.device)

    print(f"Using device: {device}")

    if device.type == "cuda":
        device = [int(device.index)] if device.index is not None else 1
    else:
        device = "cpu"
    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    save_dir_base = os.path.join(
        config['save_dir'],
        config['name']        
    )

    # setting TensorBoardLogger 
    logger = TensorBoardLogger(
        save_dir_base,  
        name=f"seed{config['seed']}_batch_size{config['data']['train_batch_size']}_epoch{config['train']['epoch']}_img2eeg_{config['lambda_img2eeg']}_img2img_{config['lambda_img2img']}_mmd_{config['lambda_mmd']}_map_{config['lambda_map']}", 
        version=opt.subjects  
    )

    os.makedirs(logger.log_dir, exist_ok=True)

    shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))
    os.makedirs(logger.log_dir, exist_ok=True)

    shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

    train_loader, val_loader, test_loader = load_data(config)

    print(f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(
            monitor='val_top1_acc',
            min_delta=0.001,     
            patience=5, 
            verbose=False,
            mode='max' 
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.001,
            patience=5,
            verbose=False,  
            mode='min' 
        )

    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=False),callbacks=[early_stop_callback, checkpoint_callback],max_epochs=config['train']['epoch'], devices=device,accelerator='cuda',logger=logger)
    print(trainer.logger.log_dir)

    ckpt_path = 'last' #None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path=ckpt_path)

    if config['exp_setting'] == 'inter-subject':
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

    with open(os.path.join(logger.log_dir,'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__=="__main__":
    main()