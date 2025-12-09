import argparse
import json
import os
import pickle as pkl
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from model import BLIP_Pretrain


parser = argparse.ArgumentParser(description='Captain Inference/Prediction Script')


parser.add_argument('--token_dict_dir', type=str, required=True, help='Directory containing token dict pickles')
parser.add_argument('--data_rna_path', type=str, required=True, help='Path to RNA test h5ad file')
parser.add_argument('--data_adt_path', type=str, required=True, help='Path to Protein/ADT test h5ad file')
parser.add_argument('--load_model_dir', type=str, required=True, help='Directory containing the trained model')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save prediction results')


parser.add_argument('--model_filename', type=str, default='pretrain_model.pt', help='Model filename')
parser.add_argument('--species', type=str, default='human', choices=['human', 'mouse'], help='Species')
parser.add_argument('--batch_size', type=int, default=1, help='Inference batch size')
parser.add_argument('--layer_size', type=int, default=128, help='Model embedding size (default: 128)')
parser.add_argument('--nlayers', type=int, default=4, help='Number of transformer layers')
parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')

args = parser.parse_args()

#
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')



def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


vocab_temp = read_json_file(os.path.join(args.token_dict_dir, 'vocab.json'))

with open(os.path.join(args.token_dict_dir, 'human_mouse_align.pickle'), 'rb') as fp:
    human_mouse_align = pkl.load(fp)
with open(os.path.join(args.token_dict_dir, 'csp_token_dict.pickle'), 'rb') as fp:
    adt_token_dict = pkl.load(fp)
with open(os.path.join(args.token_dict_dir, 'csp_align_dict.pickle'), 'rb') as fp:
    adt_align_dict = pkl.load(fp)


def preprocss_rna(data, species):
    sc.pp.filter_genes(data, min_counts=10)
    sc.pp.filter_cells(data, min_counts=200)
    if species == "mouse":
        data.var = data.var.rename(index=human_mouse_align)
        data.var_names = data.var.index
    rna_name = data.var.index.tolist()

    common_elements = set(rna_name) & set(vocab_temp.keys())


    if len(common_elements) == 0:

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


    if len(common_elements) == 0:

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
        print("adata.X contains negative values or float values, which may cause problems in the downstream analysis.")
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


def prepare_data_mouse(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    input_gene_ids_train = tokenized_train["genes"]
    input_values_train = masked_values_train
    target_values_train = tokenized_train["values"]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "adt_values": torch.tensor(adata_protein.X, dtype=torch.float32),
        "species_values": torch.ones_like(target_values_train).to(input_gene_ids_train.dtype),
        "batch_labels": torch.tensor(adata.obs["batch_id"].values, dtype=torch.long),
    }
    return train_data_pt


def prepare_data_human(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
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


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
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

# -------------------------- 主程序配置 --------------------------

hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=False,
    load_model=args.load_model_dir,
    mask_ratio=0.0,
    epochs=10,
    n_bins=51,
    MVC=False, 
    ecs_thres=0.0, 
    dab_weight=1.0,
    lr=1e-5,
    batch_size=args.batch_size,
    layer_size=args.layer_size,
    nlayers=args.nlayers, 
    nhead=args.nhead, 
    dropout=0.2, 
    schedule_ratio=0.9, 
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True, 
    include_zero_gene = False,
    freeze = False, 
    DSBN = False, 
    use_mod = True,
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

# Training/Model flags
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
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight
explicit_zero_prob = MLM and include_zero_gene 
do_sample_in_train = False and explicit_zero_prob 
per_seq_batch_sample = False

fast_transformer = config.fast_transformer
fast_transformer_backend = "flash" 
embsize = config.layer_size 
d_hid = config.layer_size 
nlayers = config.nlayers 
nhead = config.nhead 
dropout = config.dropout 
if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# Model Loading Config
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = os.path.join(args.token_dict_dir, 'args.json')
    model_file = model_dir / args.model_filename 
    vocab_file = os.path.join(args.token_dict_dir, 'vocab.json')
    
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

# -------------------------- 处理数据 --------------------------

adata = sc.read_h5ad(args.data_rna_path)

adata_protein = sc.read_h5ad(args.data_adt_path)

species = args.species
adata, adata_protein = our_step_preporcess(adata, adata_protein, species)

adata.var.set_index(adata.var.index, inplace=True)

data_is_raw = True
filter_gene_by_counts = False

adata.var["gene_name"] = adata.var.index.tolist()                 
adata_protein.var["gene_name"] = adata_protein.var.index.tolist()

adata.var["id_in_vocab"] = [
    1 if gene in vocab else -1 for gene in adata.var["gene_name"]
]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")

adata = adata[:, adata.var["id_in_vocab"] >= 0]

# Preprocessor
preprocessor = Preprocessor(
    use_key="X", 
    filter_gene_by_counts=filter_gene_by_counts, 
    filter_cell_by_counts=False, 
    normalize_total=True, 
    result_normed_key="X_normed", 
    log1p=data_is_raw, 
    result_log1p_key="X_log1p",
    subset_hvg=False, 
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins, 
    result_binned_key="X_binned", 
)

preprocessor(adata, batch_key=None)

input_layer_key = {
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
all_counts = (
    adata.layers[input_layer_key].A
    if sp.issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
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

print(f"train set number of samples: {tokenized_train['genes'].shape[0]}, feature length: {tokenized_train['genes'].shape[1]}")

if species == 'human':
    train_data_pt = prepare_data_human(sort_seq_batch=per_seq_batch_sample)
elif species == 'mouse':
    train_data_pt = prepare_data_mouse(sort_seq_batch=per_seq_batch_sample)

train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=config.batch_size,
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
)

# -------------------------- 模型初始化 --------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab) 

model = TransformerModel(
    ntokens, embsize, nhead, d_hid, nlayers,
    nlayers_cls=3, n_cls=1 if CLS else 1, vocab=vocab, dropout=dropout,
    pad_token=pad_token, pad_value=pad_value, do_mvc=MVC, do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS, num_batch_labels=1, domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style, n_input_bins=n_input_bins, cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style, ecs_threshold=ecs_threshold, explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer, fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)

if config.load_model is not None:
    try:
        rna_model_state_dict = {
            k[len('rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('rna_model.')
        }
        model.load_state_dict(rna_model_state_dict,strict=False)
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

# Freeze Params
pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
for name, para in model.named_parameters():
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        print(f"freezing weights for: {name}")
        para.requires_grad = False
post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

print(f"Total Pre freeze Params {pre_freeze_param_count}")
print(f"Total Post freeze Params {post_freeze_param_count}")

# ADT Model
print("Creating ADT model")
adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387) 
adt_model_state_dict = {
    k[len('adt_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('adt_model.')
}
adt_model.load_state_dict(adt_model_state_dict)

model = CombinedModel(model, adt_model)
model.to(device)
model.eval()

# -------------------------- 推理循环 --------------------------

true_adt_data = []
predicted_adt_data = []

wsad = 0
with torch.no_grad():
    for batch, batch_data in enumerate(train_loader):
        print(wsad)
        wsad += 1

        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)

        output_dict, transformer_out = model.rna_model(
            input_gene_ids,
            input_values,
            species_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None if INPUT_BATCH_LABELS or config.DSBN else None,
            CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS,
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
        pair.extend(labels_adt_data.cpu().squeeze().tolist())
        true_adt_data.append(pair)

        pair = []
        pair.extend(adt_to_out.cpu().squeeze().tolist())
        predicted_adt_data.append(pair)



with open(save_dir / "true_adt_data_scale.pickle", 'wb') as file:
    pkl.dump(true_adt_data, file)

with open(save_dir / "predicted_adt_scale.pickle", 'wb') as file:
    pkl.dump(predicted_adt_data, file)