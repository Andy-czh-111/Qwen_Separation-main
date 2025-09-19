# train.py

import os
import yaml
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import permutations
import argparse
import torch.distributed as dist

# --- 项目模块导入 ---
from nnet.Qwen import QwenForAudioSeparation 
from loader.datareader import DataReader
from loader.music_datareader import MusicDataReader
from nnet.reference import WavLM_feat, Encodec 
# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pit_loss(preds, targets):
    """
    计算排列不变性训练 (Permutation Invariant Training, PIT) 损失。
    这是所有分离任务（多说话人、音乐）的核心损失函数。

    Args:
        preds (list[torch.Tensor]): 模型预测的logits列表，每个元素的形状为 (B, V, T)。
                                     V是词汇表大小，T是序列长度。
        targets (list[torch.Tensor]): 目标token序列列表，每个元素的形状为 (B, T)。

    Returns:
        torch.Tensor: 该批次下，所有可能排列中的最小平均损失。
    """
    # 获取所有可能的目标排列组合
    perms = list(permutations(range(len(targets))))
    
    perm_losses = []
    
    # 对每一种排列计算总损失
    for p in perms:
        current_perm_loss = 0.0
        for i, target_idx in enumerate(p):
            # 使用交叉熵计算模型预测(logits)和目标token之间的损失
            loss = F.cross_entropy(preds[i], targets[target_idx])
            current_perm_loss += loss
        perm_losses.append(current_perm_loss / len(p)) # 对当前排列的损失取平均
        
    # 从所有排列的总损失中，选择最小的一个作为最终损失
    min_loss = torch.min(torch.stack(perm_losses))
    
    return min_loss

def collate_fn(batch):
    """
    自定义的 Collate Function，用于处理来自不同任务、结构不一的数据。
    它负责将一个批次的数据样本打包成一个规整的Tensor批次。
    """
    mixes, names, max_norms, task_ids = [], [], [], []
    targets_list = [] # targets可以是列表(语音分离)或字典(音乐分离)

    for item in batch:
        mixes.append(item['mix'].T) # 转置以适应pad_sequence
        targets_list.append(item['targets'])
        names.append(item['name'])
        max_norms.append(item['max_norm'])
        task_ids.append(item['task_id'])
    
    # 对mix进行填充，使其在批次内长度一致
    mixes_padded = nn.utils.rnn.pad_sequence(mixes, batch_first=True, padding_value=0.0).transpose(1, 2)
    
    return {
        'mix': mixes_padded,
        'targets': targets_list, # 保持原始结构，在训练循环中动态处理
        'name': names,
        'max_norm': torch.tensor(max_norms),
        'task_id': torch.tensor(task_ids, dtype=torch.long)
    }

class AudioSeparationTrainer:
    """
    通用音频分离训练器 - 整合优化版
    """
    def __init__(self, config_path: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device('cuda', rank)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("正在初始化模型及外部组件...")
        self.wavlm = WavLM_feat(self.device).wavlm
        self.codec = Encodec(self.device)
        
        self.model = QwenForAudioSeparation(
            self.config['nnet_conf'],
            self.wavlm,
            # 从codec实例动态获取词汇表大小，而不是硬编码
            xcodec_tokenizer_size=self.codec.decoder.quantizer.codebook_size 
        ).to(self.device)
        
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        
        logger.info("正在初始化数据加载器...")
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['train']['learning_rate']),
            weight_decay=float(self.config['train']['weight_decay'])
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', 
            patience=self.config['train']['lr_patience'], 
            factor=self.config['train']['lr_gamma']
        )
        self.scaler = GradScaler(enabled=self.config['train']['use_amp'])
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        if self.rank == 0: # 只有主进程负责写日志和保存模型
            self.log_dir = self.config['train']['log_dir']
            self.checkpoint_dir = self.config['train']['checkpoint_dir']
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
        
        if self.config['train'].get('resume_from'):
            self.load_checkpoint(self.config['train']['resume_from'])

    def _create_dataloaders(self):
        """根据配置文件动态创建所有任务的数据加载器"""
        train_datasets, val_datasets = [], []
        
        for task_name, task_config in self.config['tasks'].items():
            if self.rank == 0:
                logger.info(f"为任务 '{task_name}' 创建数据集...")
            
            Reader = MusicDataReader if task_config['type'] == 'music' else DataReader
            
            train_datasets.append(Reader(filename=task_config['train_scp'], **task_config['datareader_args']))
            val_datasets.append(Reader(filename=task_config['val_scp'], **task_config['datareader_args']))

        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config['train']['batch_size'], sampler=train_sampler,
            collate_fn=collate_fn, num_workers=self.config['train']['num_workers'], pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config['train']['batch_size'], sampler=val_sampler,
            collate_fn=collate_fn, num_workers=self.config['train']['num_workers'], pin_memory=True
        )
        
        return train_loader, val_loader

    def _compute_loss(self, predicted_logits, targets_audio, task_id):
        """
        动态计算损失的核心函数。
        1. 将目标音频实时编码为目标Tokens。
        2. 根据任务ID选择合适的损失计算策略（PIT或标准交叉熵）。
        """
        batch_loss = 0.0
        
        with torch.no_grad():
            target_tokens_batch = []
            for target_group in targets_audio:
                # 根据数据加载器返回的targets结构（列表或字典）来处理
                audio_list = list(target_group.values()) if isinstance(target_group, dict) else target_group
                # 使用codec将音频波形编码为离散的token序列
                tokens = [self.codec.emb2token(self.codec.get_embedding(t.unsqueeze(0).to(self.device))).squeeze(0) for t in audio_list]
                target_tokens_batch.append(tokens)

        for i in range(len(task_id)):
            current_task_id = task_id[i].item()
            task_name = self.config['task_map'][current_task_id]
            
            # 提取当前样本的预测和目标
            preds = [p[i].unsqueeze(0) for p in predicted_logits]
            targets = [t[i].unsqueeze(0) for t in target_tokens_batch[i]]
            
            # 根据任务类型决定输出头的数量和损失函数
            if task_name in ['Speaker_separation', 'music_sources_separation']:
                num_outputs = len(targets)
                loss = pit_loss(preds[:num_outputs], targets)
            else: # 默认为去噪/增强任务
                loss = F.cross_entropy(preds[0], targets[0])
            
            batch_loss += loss

        return batch_loss / len(task_id)

    def _run_epoch(self, epoch, is_train=True):
        """运行一个完整的训练或验证周期"""
        self.model.train(is_train)
        loader = self.train_loader if is_train else self.val_loader
        loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        desc = f"Epoch {epoch+1} [{'Train' if is_train else 'Valid'}]"
        progress_bar = tqdm(loader, desc=desc, disable=(self.rank != 0), dynamic_ncols=True)

        for batch in progress_bar:
            mix = batch['mix'].to(self.device, non_blocking=True)
            targets = batch['targets']
            task_id = batch['task_id'].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(is_train):
                with autocast(enabled=self.config['train']['use_amp']):
                    predicted_logits = self.model(mix, task_id)
                    loss = self._compute_loss(predicted_logits, targets, task_id)

            if is_train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1

            total_loss += loss.item()
            if self.rank == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 在所有进程上同步损失
        avg_loss_tensor = torch.tensor(total_loss / len(loader)).to(self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        
        return avg_loss_tensor.item()

    def train(self):
        """主训练循环"""
        for epoch in range(self.start_epoch, self.config['train']['num_epochs']):
            train_loss = self._run_epoch(epoch, is_train=True)
            val_loss = self._run_epoch(epoch, is_train=False)
            
            if self.rank == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best)
            
            self.scheduler.step(val_loss)

    def save_checkpoint(self, epoch, is_best):
        """保存检查点"""
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(state, filename)
        
        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
            shutil.copyfile(filename, best_filename)
            logger.info(f"已保存新的最佳模型，验证损失为: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """从检查点恢复训练"""
        if os.path.isfile(checkpoint_path):
            logger.info(f"正在从 '{checkpoint_path}' 加载检查点")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.start_epoch = checkpoint['epoch']
            self.model.module.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.best_val_loss = checkpoint['best_val_loss']
            
            logger.info(f"检查点加载完毕，将从 epoch {self.start_epoch} 继续训练。")
        else:
            logger.error(f"检查点文件未找到: '{checkpoint_path}'")

def main():
    parser = argparse.ArgumentParser(description="通用音频分离模型训练脚本")
    parser.add_argument("-c", "--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    
    trainer = AudioSeparationTrainer(args.config, rank, world_size)
    trainer.train()

if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=4 train.py --config config/train_qwen.yml