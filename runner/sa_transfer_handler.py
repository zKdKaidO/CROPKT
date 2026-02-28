import os.path as osp
import torch
import wandb
import numpy as np

from .sa_handler import SAHandler
from model.deepmil_tfl import DeepMIL_TFL_MoE
from model.utils import load_model, general_init_weight
from utils.func import fetch_kws
from utils.io import save_prediction_surv, init_layers_with_pretrain_weights
from utils.func_config import DATASET_LIST


class SATransferHandler(SAHandler):
    """
    This class handles the initialization, training, and testing 
    of SA (Survival Analysis) models for WSIs, especially for Transfer Learning.
    """
    def __init__(self, cfg):
        assert 'transfer_learning' in cfg and cfg['transfer_learning'] is True

        # two important settings for transfer learning
        self.transfer_with_patch_feat = cfg['transfer_with_patch_feat']
        self.transfer_fine_tuning = cfg['transfer_fine_tuning']

        # run setup of cuda, seed, path, model, loss, optimizer
        # LR scheduler, evaluator, and evaluation metrics with
        # the functions written to override those base ones. 
        super().__init__(cfg)
        # --- THÊM VÀO ---
        # Hàm Loss cho Domain Discriminator (Phân loại Source/Target)
        self.domain_criterion = torch.nn.BCEWithLogitsLoss()
        # ----------------
    
    # Hàm tính giá trị lambda (alpha) dựa trên tiến độ huấn luyện (p đi từ 0 đến 1)
    def get_dynamic_lambda(self, p):
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    @staticmethod
    def func_load_model(cfg):
        arch = cfg['arch']
        arch_cfg = fetch_kws(cfg, prefix=arch.lower())
        model = load_model(cfg['arch'], **arch_cfg)
        if cfg['init_wt']:
            model.apply(general_init_weight)

        if 'transfer_fine_tuning' in cfg and cfg['transfer_fine_tuning']:
            path_source_model = osp.join(cfg['transfer_load_ckpt_path'], 'train_model-last.pth')
            load_layers = ['pred_head.weight', 'pred_head.bias']
            model = init_layers_with_pretrain_weights(
                model, path_source_model, layers=load_layers
            )
            print(f"[SATransferHandler] loaded the weights of {load_layers} from {path_source_model}.")

        return model

    def save_prediction_results(self, data_cltor, path_to_save, **kws):
        save_prediction_surv(data_cltor['uid'], data_cltor['y'], data_cltor['y_hat'], path_to_save, **kws)

    """
    hàm quyết định xem đợt huấn luyện này sẽ dùng mạng MoE phức tạp hay mạng bình thường
    """
    def _update_network(self, xs, ys):
        """
        Update network using one batch data
        """
        if isinstance(self.net, DeepMIL_TFL_MoE):
            val_loss, val_preds = self._update_moe_network(xs, ys)
        else:
            val_loss, val_preds = self._update_normal_network(xs, ys)
        return val_loss, val_preds
    """
    luồng train tiêu chuẩn của PyTorch
    1. Duyệt qua Batch: for i in range(n_sample):
    2. Cho dữ liệu qua Model: pred = self.net(...) -> Lấy dự đoán
    3. Xóa Gradient cũ: self.optimizer.zero_grad()
    4. Tính Loss Tiên lượng: pred_loss = self.calc_objective_loss(bag_preds, bag_label)
    5. Lan truyền ngược & Cập nhật: pred_loss.backward() và self.optimizer.step()
    """
    def _update_normal_network(self, xs, ys):
        n_sample = len(xs)
        all_raw_pred = []

        for i in range(n_sample):
            X, ext_data = xs[i]
            if self.transfer_with_patch_feat:
                X = X.cuda()
                pred = self.net(X, ext_data.cuda())
            else:
                X = X.cuda()
                pred = self.net(X)
            all_raw_pred.append(pred)

        # 3.1 zero gradients buffer
        self.optimizer.zero_grad()

        # 3.2 loss
        bag_preds = torch.cat(all_raw_pred, dim=0) # [B, num_cls]
        bag_label = torch.cat(ys, dim=0) # [B, 2]
        pred_loss = self.calc_objective_loss(bag_preds, bag_label)

        # 3.3 backward gradients and update networks
        if isinstance(pred_loss, torch.Tensor) and pred_loss.requires_grad:
            pred_loss.backward()
            self.optimizer.step()
            self.steplr.step()
            val_loss = pred_loss.item()
        else:
            print("[batch train] warning: loss is not evaluated; skipped this batch training.")
            val_loss = 0

        val_preds = bag_preds.detach().cpu()
        return val_loss, val_preds
    """
    Hàm đc gọi khi model là DeepMIL_TFL_MoE

    """
    def _update_moe_network(self, xs, ys):
        n_sample = len(xs)
        all_raw_pred = []
        batch_router_scores = torch.zeros(self.net.n_experts)
        balance_loss, router_z_loss = .0, .0
        all_domain_preds = [] # MỚI THÊM: Mảng chứa dự đoán của cảnh sát
        all_domain_targets = [] # MỚI: Tạo mảng chứa nhãn miền THẬT

        for i in range(n_sample):
            # --- [SỬA ĐOẠN NÀY] ---
            # xs[i] bây giờ trả về: (transfer_feat, (original_feats, domain_label))
            X, packed_extra = xs[i] 
            
            # Tách gói dữ liệu ra
            original_feats, domain_label = packed_extra
            
            # Lưu nhãn miền thật lại để tí nữa tính Loss
            all_domain_targets.append(domain_label)
            # ----------------------

            if self.transfer_with_patch_feat:
                X = X.cuda()
                # Chỉ đưa original_feats vào model (không đưa domain_label vào đây gây lỗi)
                pred, router_scores, cur_balance_loss, cur_router_z_loss, domain_preds = self.net(X, original_feats.cuda())
            else:
                X = X.cuda()
                pred, router_scores, cur_balance_loss, cur_router_z_loss, domain_preds = self.net(X)
            
            all_domain_preds.append(domain_preds) # MỚI THÊM: Gom kết quả lại

            all_raw_pred.append(pred) # Lưu dự đoán sống/chết
            batch_router_scores += router_scores.cpu().squeeze(0) # Lưu điểm Router để log
            balance_loss += cur_balance_loss # Cộng dồn Loss phụ 1
            router_z_loss += cur_router_z_loss # Cộng dồn Loss phụ 2

        # 3.1 zero gradients buffer
        self.optimizer.zero_grad()

        # 3.2 loss
        bag_preds = torch.cat(all_raw_pred, dim=0) # [B, num_cls]
        bag_label = torch.cat(ys, dim=0) # [B, 2]
        pred_loss = self.calc_objective_loss(bag_preds, bag_label)
        batch_router_scores = batch_router_scores / n_sample
        self._wandb_log_router_scores(batch_router_scores)
        aux_loss = self.cfg['loss_balance_weight'] * balance_loss / n_sample
        aux_loss += self.cfg['loss_router_z_weight'] * router_z_loss / n_sample
        wandb.log({'train/aux_balance_loss': balance_loss.item() / n_sample})
        wandb.log({'train/aux_router_z_loss': router_z_loss.item() / n_sample})
        wandb.log({'train/aux_loss': aux_loss.item()})

        # --- BẮT ĐẦU PHẦN DOMAIN ADAPTATION ---
        bag_domain_preds = torch.cat(all_domain_preds, dim=0)
        
        # TẠO NHÃN MIỀN (DOMAIN LABELS):
        # GIẢ ĐỊNH TẠM THỜI: Toàn bộ batch này là Source (Nhãn 1)
        # (Sau này khi DataLoader trộn 2 miền, bạn sẽ lấy nhãn thật từ data)
        target_domain = torch.cat(all_domain_targets, dim=0).to(bag_domain_preds.device)
        
        # Tính Domain Loss (L_d)
        domain_loss = self.domain_criterion(bag_domain_preds, target_domain)
        
        # Tính Lambda (Lịch trình thích nghi)
        # Giả lập tham số tiến độ p (cần lấy số epoch hiện tại, tạm để 0.5 để chạy thử)
        p = 0.5 
        alpha = self.get_dynamic_lambda(p)
        
        # Log Domain Loss lên Wandb để theo dõi
        wandb.log({'train/domain_loss': domain_loss.item(), 'train/lambda_alpha': alpha})
        
        # TỔNG HỢP LOSS (Đã cộng cả Cảnh sát vào)
        pred_loss = pred_loss + aux_loss + (alpha * domain_loss)
        # --------------------------------------

        # 3.3 backward gradients and update networks
        if isinstance(pred_loss, torch.Tensor) and pred_loss.requires_grad:
            pred_loss.backward()
            self.optimizer.step()
            self.steplr.step()
            val_loss = pred_loss.item()
        else:
            print("[batch train] warning: loss is not evaluated; skipped this batch training.")
            val_loss = 0

        val_preds = bag_preds.detach().cpu()
        return val_loss, val_preds

    def _wandb_log_router_scores(self, batch_router_scores):
        if 'transfer_self_feat' in self.cfg and self.cfg['transfer_self_feat']:
            dataset_order = [self.cfg['dataset_name']]
        else:
            dataset_order = []
        
        if 'transfer_feat_idx' in self.cfg and self.cfg['transfer_feat_idx'] is not None:
            for i in self.cfg['transfer_feat_idx']:
                dataset_order.append(DATASET_LIST[i])

        assert len(dataset_order) == len(batch_router_scores), "Found a wrong dataset order."
        log_data = dict()
        for i, d in enumerate(dataset_order):
            log_data[f"train/router_score/expert_{d[5:]}"] = batch_router_scores[i].item()

        wandb.log(log_data)

    def test_model(self, model, loader, loader_name, ckpt_path=None, **kws):
        if ckpt_path is not None:
            net_ckpt = torch.load(ckpt_path)
            model.load_state_dict(net_ckpt['model'], strict=False)
        model.eval()

        all_idx, all_raw_pred, all_pred, all_gt = [], [], [], []
        for data_idx, data_x, data_y in loader:
            # data_x = (feats, coords) | data_y = label_slide
            X, ext_data = data_x
            data_label = data_y
            with torch.no_grad():
                if self.transfer_with_patch_feat:
                    X = X.cuda()
                    raw_pred = model(X, ext_data.cuda())
                else:
                    X = X.cuda()
                    raw_pred = model(X)
                # To convert raw predictions for 
                # evaluation and prediction saving
                pred = self.output_converter(raw_pred)
            all_gt.append(data_label)
            all_raw_pred.append(raw_pred.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_idx.append(data_idx)
        
        all_raw_pred = torch.cat(all_raw_pred, dim=0) # [B, num_cls]
        all_pred = torch.cat(all_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0) # [B, 2]
        all_idx = torch.cat(all_idx, dim=0).squeeze() # [B, ]

        cltor = dict()
        all_uids = self._get_unique_id(loader_name, all_idx)
        cltor['pred'] = {'y': all_gt, 'raw_y_hat': all_raw_pred, 'y_hat': all_pred, 'uid': all_uids, 'name': loader_name}

        return cltor
