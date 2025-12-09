import argparse
import copy
import gc
import glob
import itertools
import json
import os
import pickle as pkl
import random
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import matplotlib.pyplot as plt
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import scipy.stats as st
import seaborn as sns
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from tqdm import tqdm

# scGPT imports
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    quantile_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from performer_pytorch import BLIP_Pretrain

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="scGPT Fine-tuning and Inference")

    # Paths
    parser.add_argument("--data_dir", type=str, default="/home/jiboya/Captain/cell_type_anno/dataset1/", help="Directory containing dataset files")
    parser.add_argument("--save_dir", type=str, default="/home/jiboya/Captain/cell_type_anno/dataset1/", help="Directory to save results")
    parser.add_argument("--vocab_file", type=str, default="/home/jiboya/Captain/pretrain/vocab.json", help="Path to vocab json")
    parser.add_argument("--token_dict_dir", type=str, default="/home/jiboya/scBLIP/token_dict/", help="Directory for token dictionaries")
    parser.add_argument("--load_model", type=str, default="/pool2/jiboya/captain_model", help="Path to pretrained model directory")
    
    # Files
    parser.add_argument("--train_rna_file", type=str, default="pbmc_gene_downsampled_train.h5ad")
    parser.add_argument("--train_adt_file", type=str, default="pbmc_protein_downsampled_train.h5ad")
    parser.add_argument("--test_rna_file", type=str, default="pbmc_gene_downsampled_test.h5ad")
    parser.add_argument("--test_adt_file", type=str, default="pbmc_protein_downsampled_test.h5ad")

    # Settings
    parser.add_argument("--species", type=str, default="human", choices=["human", "mouse"])
    parser.add_argument("--gpu_device", type=str, default="6", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=26)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze", action="store_true", default=False)
    
    return parser.parse_args()

args = get_args()

# Environment Settings
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

# -----------------------------------------------------------------------------
# Utils & Data Loading
# -----------------------------------------------------------------------------

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Load Dictionaries
vocab_temp = read_json_file(args.vocab_file)

with open(os.path.join(args.token_dict_dir, 'human_mouse_align.pickle'), 'rb') as fp:
    human_mouse_align = pkl.load(fp)
with open(os.path.join(args.token_dict_dir, 'adt_token_dict.pickle'), 'rb') as fp:
    adt_token_dict = pkl.load(fp)
with open(os.path.join(args.token_dict_dir, 'adt_align_dict.pickle'), 'rb') as fp:
    adt_align_dict = pkl.load(fp)

def preprocss_rna(data, species):
    sc.pp.filter_genes(data, min_counts=10)
    sc.pp.filter_cells(data, min_counts=200)
    if species == "mouse":
        data.var = data.var.rename(index=human_mouse_align)
        data.var_names = data.var.index
    rna_name = data.var.index.tolist()

    common_elements = set(rna_name) & set(vocab_temp.keys())

    print("Presence of RNA genes in AnnData object:", len(common_elements))
    if len(common_elements) == 0:
        print("No matching genes found, exiting.")
        sys.exit()

    return data

def preprocss_adt(data, species):
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)
    sc.pp.scale(data)
    data.var = data.var.rename(index=adt_align_dict)
    data.var_names = data.var.index

    duplicated_genes = data.var_names.duplicated(keep='first')
    genes_to_keep = ~duplicated_genes
    data = data[:, genes_to_keep]
    
    gene_name = list(adt_token_dict.keys())
    adt_name = data.var.index.tolist()

    common_elements = set(adt_name) & set(gene_name)

    print("Presence of ADT genes in AnnData object:", len(common_elements))
    if len(common_elements) == 0:
        print("No matching proteins found, exiting.")
        sys.exit()

    new_adata = sc.AnnData(np.zeros((data.shape[0], len(gene_name))), obs=data.obs.copy(), var=pd.DataFrame(index=gene_name))

    for gene in common_elements:
        if sp.issparse(data.X):
            try:
                new_adata.X[:, new_adata.var_names == gene] = data.X[:, data.var_names == gene].toarray()
            except IndexError:
                print(f"IndexError when processing {gene}")
                continue
        else:
            try:
                new_adata.X[:, new_adata.var_names == gene] = data.X[:, data.var_names == gene]
            except IndexError:
                print(f"IndexError when processing {gene}")
                continue

    return new_adata

def check_adata_x(adata): 
    if sp.issparse(adata.X): 
        non_zero_data = adata.X.data 
        has_negative = (non_zero_data < 0).any() 
        has_float = (non_zero_data != non_zero_data.astype(int)).any() 
    else: 
        has_negative = (adata.X < 0).any() 
        has_float = (adata.X != adata.X.astype(int)).any()
    if has_negative or has_float: 
        print("adata.X contains negative values or float values, which may cause problems in the downstream analysis.")
        sys.exit()

def our_step_preporcess(adata, adata_protein, species):
    check_adata_x(adata)
    # check_adata_x(adata_protein) # Often ADT is float/normalized, check constraints if needed
    rna_data_pre = preprocss_rna(adata, species=species)
    adt_data_pre = preprocss_adt(adata_protein, species=species)
    common_obs = rna_data_pre.obs_names.intersection(adt_data_pre.obs_names)
    rna_data_pre = rna_data_pre[common_obs]
    adt_data_pre = adt_data_pre[common_obs]
    return rna_data_pre, adt_data_pre

def prepare_data(tokenized_data, adata_protein, celltypes_labels, species="human", sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values = random_mask_value(
        tokenized_data["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    print(f"{(masked_values == mask_value).sum() / (masked_values - pad_value).count_nonzero():.4f}")

    input_gene_ids = tokenized_data["genes"]
    input_values = masked_values
    target_values = tokenized_data["values"]
    
    species_val = torch.ones_like(target_values).to(input_gene_ids.dtype) if species == "mouse" else torch.zeros_like(target_values).to(input_gene_ids.dtype)

    data_pt = {
        "gene_ids": input_gene_ids,
        "values": input_values,
        "target_values": target_values,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": species_val,
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    return data_pt

# Dataset Class
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# DataLoader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = False
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

def train(model: nn.Module, loader: DataLoader) -> None:
    model.train()
    total_loss = 0.0
    start_time = time.time()
    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        # target_values = batch_data["target_values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
        celltype_labels = batch_data["celltype_labels"].to(device)

        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict, transformer_out = model.rna_model(
                input_gene_ids,
                input_values,
                species_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
            )

            adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten, labels_adt_data, adt_mask = model.adt_model(
                adt_data,
                transformer_out,
                src_key_padding_mask,
                adt_values,
                output_atten=False
            )

            loss = criterion_cls(celltype_mlp(adt_embeddings[:,-1,:]), celltype_labels)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    f"can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr_curr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f"lr {lr_curr:05.4f} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f}")
            total_loss = 0
            start_time = time.time()

def test(model: nn.Module, loader: DataLoader) -> None:
    model.eval()
    total_loss = 0.0
    predictions = []
    embs = []
    
    with torch.no_grad():
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            species_values = batch_data["species_values"].to(device)
            adt_values = batch_data["adt_values"].to(device)
            adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
            celltype_labels = batch_data["celltype_labels"].to(device)

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict, transformer_out = model.rna_model(
                    input_gene_ids,
                    input_values,
                    species_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=do_sample_in_train,
                )

                adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten, labels_adt_data, adt_mask = model.adt_model(
                    adt_data,
                    transformer_out,
                    src_key_padding_mask,
                    adt_values,
                    output_atten=False
                )
                output_values = celltype_mlp(adt_embeddings[:,-1,:])
                
                loss = criterion_cls(output_values, celltype_labels)

            total_loss += loss.item()
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)
            embs.append(adt_embeddings[:,-1,:])
        
        print(total_loss)
        return np.concatenate(predictions, axis=0), embs

class Config:
    def __init__(self, defaults):
        for key, value in defaults.items():
            setattr(self, key, value)

class CombinedModel(nn.Module):
    def __init__(self, main_model, sub_model):
        super(CombinedModel, self).__init__()
        self.rna_model = main_model
        self.adt_model = sub_model

    def forward(self, x):
        pass

class Identity_Celltype(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity_Celltype, self).__init__()
        self.fc1 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
hyperparameter_defaults = dict(
    seed=args.seed,
    dataset_name="ms",
    do_train=True,
    load_model=args.load_model,
    mask_ratio=0.0,
    epochs=args.epochs,
    n_bins=51,
    MVC=False, 
    ecs_thres=0.0, 
    dab_weight=1.0,
    lr=args.lr,
    batch_size=args.batch_size,
    layer_size=512,
    nlayers=12,
    nhead=8,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=args.freeze,
    DSBN=False,
    use_mod=True,
)

config = Config(hyperparameter_defaults)
print(config)
set_seed(config.seed)

# Input/Preprocessing Settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"
include_zero_gene = config.include_zero_gene
max_seq_len = 3001
n_bins = config.n_bins
input_style = "binned"
output_style = "binned"

# Training Settings
MLM = False
CLS = False
ADV = False
CCE = False
MVC = config.MVC
ECS = config.ecs_thres > 0
DAB = False
INPUT_BATCH_LABELS = False
input_emb_style = "continuous"
cell_emb_style = "cls"
adv_E_delay_epochs = 0
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight
explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob
per_seq_batch_sample = False

lr = config.lr
lr_ADV = 1e-3
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# Model Settings
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"
embsize = config.layer_size
d_hid = config.layer_size
nlayers = config.nlayers
nhead = config.nhead
dropout = config.dropout

log_interval = 100
save_eval_interval = config.save_eval_interval
do_eval_scib_metrics = True

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

DAB_separate_optim = True if DAB > 1 else False

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "CAPTAIN_Base.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(f"Resume model from {model_file}")
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)

model = TransformerModel(
    ntokens, embsize, nhead, d_hid, nlayers, nlayers_cls=3, n_cls=1 if CLS else 1,
    vocab=vocab, dropout=dropout, pad_token=pad_token, pad_value=pad_value,
    do_mvc=MVC, do_dab=DAB, use_batch_labels=INPUT_BATCH_LABELS, num_batch_labels=1,
    domain_spec_batchnorm=config.DSBN, input_emb_style=input_emb_style,
    n_input_bins=n_input_bins, cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style, ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob, use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend, pre_norm=config.pre_norm,
)

if config.load_model is not None:
    try:
        rna_model_state_dict = {
            k[len('module.rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.rna_model')
        }
        model.load_state_dict(rna_model_state_dict)
        print(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

for name, para in model.named_parameters():
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        print(f"freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
print(f"Total Pre freeze Params {pre_freeze_param_count}")
print(f"Total Post freeze Params {post_freeze_param_count}")

print("Creating model")
adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387)
adt_model_state_dict = {
    k[len('module.adt_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.adt_model')
}
adt_model.load_state_dict(adt_model_state_dict)

model = CombinedModel(model, adt_model)
model.to(device)

if ADV:
    discriminator = AdversarialDiscriminator(d_model=embsize, n_cls=1).to(device)

criterion = masked_mse_loss
criterion_quantile = quantile_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=config.schedule_ratio)

if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(optimizer_dab, schedule_interval, gamma=config.schedule_ratio)
if ADV:
    criterion_adv = nn.CrossEntropyLoss()
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, schedule_interval, gamma=config.schedule_ratio)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, schedule_interval, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# -----------------------------------------------------------------------------
# Data Loading & Processing Function
# -----------------------------------------------------------------------------
def process_and_load_data(rna_path, adt_path, vocab, species, is_train=True):
    print(f"Loading {'Train' if is_train else 'Test'} Data...")
    adata = sc.read_h5ad(rna_path)
    adata_protein = sc.read_h5ad(adt_path)
    
    # Preprocess
    adata, adata_protein = our_step_preporcess(adata, adata_protein, species)
    
    adata.var.set_index(adata.var.index, inplace=True)
    data_is_raw = True
    adata.var["gene_name"] = adata.var.index.tolist()
    adata_protein.var["gene_name"] = adata_protein.var.index.tolist()

    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    celltype_id_labels = adata.obs["celltype.l2"].astype("category").cat.codes.values
    num_types = len(np.unique(celltype_id_labels))
    adata.obs["celltype_id"] = celltype_id_labels
    celltypes_labels = np.array(adata.obs["celltype_id"].tolist())

    # Preprocessor
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=True,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key=None)

    input_layer_key = {"normed_raw": "X_normed", "log1p": "X_normed", "binned": "X_binned"}[input_style]
    all_counts = adata.layers[input_layer_key].A if sp.issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key]
    genes = adata.var["gene_name"].tolist()
    
    gene_ids = np.array(vocab(genes), dtype=int)
    
    tokenized_data = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    
    print(f"Number of samples: {tokenized_data['genes'].shape[0]}, feature length: {tokenized_data['genes'].shape[1]}")
    
    data_pt = prepare_data(tokenized_data, adata_protein, celltypes_labels, species=species)
    
    loader = prepare_dataloader(
        data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    return loader, num_types

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

# Load Train Data
train_rna_path = os.path.join(args.data_dir, args.train_rna_file)
train_adt_path = os.path.join(args.data_dir, args.train_adt_file)
train_loader, num_types_train = process_and_load_data(train_rna_path, train_adt_path, vocab, args.species, is_train=True)

# Load Test Data
test_rna_path = os.path.join(args.data_dir, args.test_rna_file)
test_adt_path = os.path.join(args.data_dir, args.test_adt_file)
test_loader, num_types_test = process_and_load_data(test_rna_path, test_adt_path, vocab, args.species, is_train=False)

# Celltype MLP
celltype_mlp = Identity_Celltype(dropout=0., h_dim=256, out_dim=num_types_train).to(device)

# Training Loop
for epoch in range(1, epochs + 1):
    if config.do_train:
        train(model, loader=train_loader)
        predictions, embs = test(model, loader=test_loader)

        name = str(save_dir) + "/" + str(epoch) + "finetune_model.pt"
        torch.save(model.state_dict(), name)
        name = str(save_dir) + "/" + str(epoch) + "predictions.npy"
        np.save(name, predictions)
        name = str(save_dir) + "/" + str(epoch) + "embs.pt"
        torch.save([emb.cpu() for emb in embs], name) 

        scheduler.step()
        if DAB_separate_optim:
            scheduler_dab.step()
        if ADV:
            scheduler_D.step()
            scheduler_E.step()
