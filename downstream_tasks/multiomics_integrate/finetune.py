import argparse
import json
import os
import pickle as pkl
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from sklearn import preprocessing 

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.loss import (
    masked_mse_loss,
    quantile_loss,
)
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed
from model import BLIP_Pretrain


parser = argparse.ArgumentParser(description='Captain Training Script')

parser.add_argument('--token_dict_dir', type=str, required=True, help='Directory containing token dict pickles')
parser.add_argument('--data_rna_path', type=str, required=True, help='Path to RNA h5ad file')
parser.add_argument('--data_adt_path', type=str, required=True, help='Path to Protein/ADT h5ad file')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save outputs')
parser.add_argument('--load_model_dir', type=str, default=None, help='Directory containing pre-trained model')
parser.add_argument('--model_filename', type=str, default="CAPTAIN_Base.pt", help='Name of the model file to load')
parser.add_argument('--prior_know', type=str, default=None, help='Directory containing prior knowledge file')

parser.add_argument('--species', type=str, default='human', choices=['human', 'mouse'], help='Species (human or mouse)')
parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--freeze', action='store_true', help='Freeze encoder weights')
# Added batch_col argument
parser.add_argument("--batch_col", type=str, default="batch", help="Column in adata.obs representing batch/donor")

args = parser.parse_args()


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


# Modified to accept batch_ids
def prepare_data_mouse(batch_ids, sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
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
        "species_values": torch.ones_like(target_values_train).to(input_gene_ids_train.dtype),
        "batch_labels": torch.from_numpy(batch_ids).long(), # Added batch_labels
    }
    return train_data_pt

# Modified to accept batch_ids
def prepare_data_human(batch_ids, sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
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
        "batch_labels": torch.from_numpy(batch_ids).long(), # Added batch_labels
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
    
    # dataset_sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        # sampler=dataset_sampler
    )
    return data_loader



def train(model: nn.Module, loader: DataLoader) -> None:
    model.train()
    # dist.barrier()
    total_loss = 0.0
    start_time = time.time()
    
    total_cls = 0.0
    total_cce = 0.0

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)
        # Added batch_labels retrieval
        batch_labels = batch_data["batch_labels"].to(device)

        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict, transformer_out = model.rna_model(
                input_gene_ids,
                input_values,
                species_values,
                src_key_padding_mask=src_key_padding_mask,
                # Passed batch_labels to the model
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
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

            loss = 0.0
            loss_weights = {
                "adt_mse": 0.8,
                "adt_quantile": 0.2,
            }

            loss_adt_mse = loss_weights["adt_mse"] * criterion(adt_to_out.squeeze(-1), labels_adt_data, adt_mask)
            loss_adt_quantile = loss_weights["adt_quantile"] * criterion_quantile(adt_to_out_quantiles, labels_adt_data, adt_mask)
            loss = loss_adt_mse + loss_adt_quantile

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
                logger.warning(f"Found infinite gradient. Current scale: {scaler.get_scale()}")
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls += loss_adt_mse.item()
        total_cce += loss_adt_quantile.item()

        if batch % log_interval == 0 and batch > 0:
            lr_curr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_cls = total_cls / log_interval
            cur_cce = total_cce / log_interval

            print(
                f"lr {lr_curr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
            )
            total_loss = 0
            total_cls = 0
            total_cce = 0
            start_time = time.time()



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

def seed_all(seed_value, cuda_deterministic=False):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=True,
    load_model=args.load_model_dir, 
    mask_ratio=0.0,
    epochs=args.epoch, 
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

# set_seed(config.seed)
# local_rank = int(os.environ.get('LOCAL_RANK', 0))
# dist.init_process_group(backend='gloo')
# torch.cuda.set_device(local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed_all(config.seed + torch.distributed.get_rank())

# Input / Preprocessing Settings
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
# Changed to True to enable batch labels usage
INPUT_BATCH_LABELS = True 
input_emb_style = "continuous"
cell_emb_style = "cls"
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight
explicit_zero_prob = MLM and include_zero_gene
do_sample_in_train = False and explicit_zero_prob
per_seq_batch_sample = False
lr = config.lr
lr_ADV = 1e-3
batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"
embsize = config.layer_size
d_hid = config.layer_size
nlayers = config.nlayers
nhead = config.nhead
dropout = config.dropout
log_interval = 100
DAB_separate_optim = True if DAB > 1 else False

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


species = args.species 

adata = sc.read_h5ad(args.data_rna_path)
adata.var_names_make_unique()
adata_protein = sc.read_h5ad(args.data_adt_path)

adata, adata_protein = our_step_preporcess(adata, adata_protein, species)

# Batch Processing
le = preprocessing.LabelEncoder()
encoded_batch = le.fit_transform(adata.obs[args.batch_col].values)
adata.obs["batch_id"] = encoded_batch
batch_ids = np.array(adata.obs["batch_id"].tolist())
num_batch_types = len(set(batch_ids))

adata.var.set_index(adata.var.index, inplace=True)
data_is_raw = True
adata.var["gene_name"] = adata.var.index.tolist()
adata_protein.var["gene_name"] = adata_protein.var.index.tolist()

adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
print(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
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

input_layer_key = "X_binned" if input_style == "binned" else "X_normed"
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

print(f"train set number of samples: {tokenized_train['genes'].shape[0]}, feature length: {tokenized_train['genes'].shape[1]}")

if species == "human":
    train_data_pt = prepare_data_human(batch_ids, sort_seq_batch=per_seq_batch_sample)
elif species == "mouse":
    train_data_pt = prepare_data_mouse(batch_ids, sort_seq_batch=per_seq_batch_sample)

train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=batch_size,
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
)

ntokens = len(vocab)
model = TransformerModel(
    ntokens, embsize, nhead, d_hid, nlayers,
    nlayers_cls=3, n_cls=1, vocab=vocab, dropout=dropout,
    pad_token=pad_token, pad_value=pad_value,
    do_mvc=MVC, do_dab=DAB, 
    # Enable batch labels and pass number of types
    use_batch_labels=True,
    num_batch_labels=num_batch_types, 
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style, n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style, mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold, explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer, fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm, prior_know=args.prior_know,
)

if config.load_model is not None:
    try:
        rna_model_state_dict = {
            k[len('module.rna_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.rna_model.')
        }
        model.load_state_dict(rna_model_state_dict, strict=False)
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

# ADT Model
print("Creating ADT model")
adt_model = BLIP_Pretrain(num_tokens2=387, adt_max_seq_len=387)
if config.load_model is not None:
    adt_model_state_dict = {
        k[len('module.adt_model.'):]: v for k, v in torch.load(model_file, map_location=device).items() if k.startswith('module.adt_model.')
    }
    adt_model.load_state_dict(adt_model_state_dict)

model = CombinedModel(model, adt_model)
model.to(device)
# model = DDP(model, device_ids=[local_rank])

# Optimizer & Scheduler
criterion = masked_mse_loss
criterion_quantile = quantile_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=config.schedule_ratio)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(optimizer_dab, schedule_interval, gamma=config.schedule_ratio)
if ADV:
    discriminator = AdversarialDiscriminator(d_model=embsize, n_cls=1).to(device)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, schedule_interval, gamma=config.schedule_ratio)
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(optimizer_E, schedule_interval, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


for epoch in range(1, epochs + 1):
    if config.do_train:
        train(model, loader=train_loader)
        
        name = save_dir / "pretrain_model.pt"
        torch.save(model.state_dict(), name)

        scheduler.step()
        if DAB_separate_optim:
            scheduler_dab.step()
        if ADV:
            scheduler_D.step()
            scheduler_E.step()
