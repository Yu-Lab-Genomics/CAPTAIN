import argparse
import json
import os
import pickle as pkl
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

# scGPT & Model Imports
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, quantile_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed
from performer_pytorch import BLIP_Pretrain

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="scGPT + BLIP Inference")

    # Paths
    parser.add_argument("--data_dir", type=str, default="/home/jiboya/Captain/multiomics/dataset2/", help="Directory containing dataset files")
    parser.add_argument("--save_dir", type=str, default="/home/jiboya/Captain/multiomics/dataset2/captain/", help="Directory to save embeddings")
    parser.add_argument("--vocab_file", type=str, default="/home/jiboya/Captain/pretrain/vocab.json", help="Path to vocab json")
    parser.add_argument("--token_dict_dir", type=str, default="/home/jiboya/scBLIP/token_dict/", help="Directory for token dictionaries")
    parser.add_argument("--load_model", type=str, default="/pool2/jiboya/captain_model/", help="Path to pretrained model directory")
    parser.add_argument('--prior_know', type=str, default=None, help='Directory containing prior knowledge file')

    # Files
    parser.add_argument("--rna_file", type=str, default="adata.h5ad", help="RNA h5ad filename")
    parser.add_argument("--adt_file", type=str, default="adata_protein.h5ad", help="ADT h5ad filename")

    # Settings
    parser.add_argument("--species", type=str, default="human", choices=["human", "mouse"])
    parser.add_argument("--gpu_device", type=str, default="7", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    
    return parser.parse_args()

args = get_args()

# Environment Settings
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

# -----------------------------------------------------------------------------
# Utilities & Processing
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
    if species == "mouse":
        data.var = data.var.rename(index=human_mouse_align)
        data.var_names = data.var.index 
    rna_name = data.var.index.tolist()

    common_elements = set(rna_name) & set(vocab_temp.keys())

    print("RNA genes present in AnnData:", len(common_elements))
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

    print("ADT proteins present in AnnData:", len(common_elements))
    if len(common_elements) == 0:
        print("No matching proteins found, exiting.")
        sys.exit()
    
    new_adata = ad.AnnData(np.zeros((data.shape[0], len(gene_name))), obs=data.obs.copy(), var=pd.DataFrame(index=gene_name))

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
        print("adata.X contains negative or float values.")
        sys.exit()

def our_step_preporcess(adata, adata_protein, species):
    check_adata_x(adata)
    check_adata_x(adata_protein)
    rna_data_pre = preprocss_rna(adata, species=species)
    adt_data_pre = preprocss_adt(adata_protein, species=species)
    common_obs = rna_data_pre.obs_names.intersection(adt_data_pre.obs_names)
    rna_data_pre = rna_data_pre[common_obs]
    adt_data_pre = adt_data_pre[common_obs]
    return rna_data_pre, adt_data_pre

def prepare_data_human(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}")

    input_gene_ids_train = tokenized_train["genes"]
    input_values_train = masked_values_train
    target_values_train = tokenized_train["values"]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": torch.zeros_like(target_values_train).to(input_gene_ids_train.dtype),
    }
    return train_data_pt

# Dataset & DataLoader
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def prepare_dataloader(data_pt: Dict[str, torch.Tensor], batch_size: int, shuffle: bool = False, intra_domain_shuffle: bool = False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
    dataset = SeqDataset(data_pt)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

class CombinedModel(nn.Module):
    def __init__(self, main_model, sub_model):
        super(CombinedModel, self).__init__()
        self.rna_model = main_model
        self.adt_model = sub_model
    def forward(self, x):
        pass

class Config:
    def __init__(self, defaults):
        for key, value in defaults.items():
            setattr(self, key, value)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
hyperparameter_defaults = dict(
    seed=args.seed,
    dataset_name="ms",
    do_train=False,
    load_model=args.load_model,
    mask_ratio=0.0,
    epochs=args.epochs,
    n_bins=51,
    MVC=True, 
    ecs_thres=0.0, 
    dab_weight=1.0,
    lr=args.lr,
    batch_size=args.batch_size,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=False,
    DSBN=False,
    use_mod=True,
)

config = Config(hyperparameter_defaults)
print(config)
set_seed(config.seed)

# Settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"
include_zero_gene = config.include_zero_gene
max_seq_len = 3001
n_bins = config.n_bins
input_style = "binned"
output_style = "binned"
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
# Model & Data Load
# -----------------------------------------------------------------------------
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "cmca00064-GSM5631553_model.pt"
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

# Process Data
file1 = os.path.join(args.data_dir, args.rna_file)
adata = sc.read_h5ad(file1)
file2 = os.path.join(args.data_dir, args.adt_file)
adata_protein = sc.read_h5ad(file2)
adata_protein.var.index = adata_protein.var.index.str.replace("AB_", "")
adata, adata_protein = our_step_preporcess(adata, adata_protein, args.species)

adata.var.set_index(adata.var.index, inplace=True)
data_is_raw = True
adata.var["gene_name"] = adata.var.index.tolist()
adata_protein.var["gene_name"] = adata_protein.var.index.tolist()

adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary.")
adata = adata[:, adata.var["id_in_vocab"] >= 0]

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
train_data = all_counts

if config.load_model is None:
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
print(f"samples: {tokenized_train['genes'].shape[0]}, feature length: {tokenized_train['genes'].shape[1]}")

train_data_pt = prepare_data_human(sort_seq_batch=per_seq_batch_sample)
train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=batch_size,
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
)

# Initialize Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)
model = TransformerModel(
    ntokens, embsize, nhead, d_hid, nlayers,
    nlayers_cls=3, n_cls=1, vocab=vocab, dropout=dropout,
    pad_token=pad_token, pad_value=pad_value,
    do_mvc=MVC, do_dab=DAB, use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=1, domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style, n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style, mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold, explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer, fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm, prior_know=args.prior_know,
)

if config.load_model is not None:
    try:
        rna_model_state_dict = {
            k[len('rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('rna_model.')
        }
        model.load_state_dict(rna_model_state_dict)
        print(f"Loading all model params from {model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

print("Creating model")
adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387)
adt_model_state_dict = {
    k[len('adt_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('adt_model.')
}
adt_model.load_state_dict(adt_model_state_dict)

model = CombinedModel(model, adt_model)
model.to(device)

if ADV:
    discriminator = AdversarialDiscriminator(d_model=embsize, n_cls=1).to(device)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# -----------------------------------------------------------------------------
# Inference Loop
# -----------------------------------------------------------------------------
model.eval()

rna_emb = []
adt_emb = []
wsad = 0

with torch.no_grad():
    for batch, batch_data in enumerate(train_loader):
        print(wsad)
        wsad += 1

        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)

        output_dict, transformer_out = model.rna_model(
            input_gene_ids,
            input_values,
            species_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
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
        
        pair = []
        pair.extend(output_dict['cell_emb'].cpu().squeeze().tolist())
        rna_emb.append(pair)

        pair = []
        pair.extend(adt_embeddings[:,-1,:].cpu().squeeze().tolist())
        adt_emb.append(pair)

# Save Results
with open(os.path.join(args.save_dir, "rna_embeddings.pickle"), 'wb') as file:
    pkl.dump(rna_emb, file)
with open(os.path.join(args.save_dir, "adt_embeddings.pickle"), 'wb') as file:
    pkl.dump(adt_emb, file)