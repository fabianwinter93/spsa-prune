from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import torch
import numpy as np

import os

from scipy.stats.qmc import Sobol, discrepancy
from scipy.stats import norm, truncnorm

import matplotlib.pyplot as plt

#REPO_ID = "mayflowergmbh/Wiedervereinigung-7b-dpo-laser"
REPO_ID = "VAGOsolutions/SauerkrautLM-1.5b"
#REPO_ID = "VAGOsolutions/SauerkrautLM-gemma-2-2b-it"
#REPO_ID = "HuggingFaceTB/SmolLM-135M"


config = AutoConfig.from_pretrained(REPO_ID)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

#config.num_hidden_layers = 1
#config.hidden_size = 256
#config.intermediate_size = 896
#config.num_attention_heads = 8

#model = AutoModelForCausalLM.from_config(config)
model = AutoModelForCausalLM.from_pretrained(REPO_ID, device_map="auto", config=config, local_files_only=True)


def get_noise_sobol(shape, seed):
    
    sobol = Sobol(d=1, seed=seed)

    m = int(np.ceil(np.log2(shape[0]*shape[1])))

    points = norm.ppf(sobol.random_base2(m), 0., 1.)
    
    points = points[:shape[0]*shape[1]]
    points = points.reshape(shape)
    
    return points

def get_noise_normal(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape)
    
def get_noise_uniform(shape, seed):
    rng = np.random.default_rng(seed)
    ret = rng.uniform(-1, 1, size=shape)
    return ret

def get_noise_truncnorm(shape, seed):
    return truncnorm.rvs(-0.5, 0.5, 0, 1, size=shape, random_state=seed)


get_noise = get_noise_normal


stored_inputs = {}
stored_inputs_bool = False


def get_save_inputs_hook(name):

    def save_inputs_hook(mdl, args, output):
        global stored_inputs
        global stored_inputs_bool

        if stored_inputs_bool:

            inputs = args[0]

            stored_inputs[name] = inputs

            return None

    return save_inputs_hook


with torch.no_grad():
    for name, mdl in model.named_modules():
        mlp_or_attn = ("self_attn" in name or "mlp" in name) and "_proj" in name
        emb_or_head = "embed_tokens" in name or "lm_head" in name

        if (is_weight := hasattr(mdl, "weight")) and not emb_or_head and mlp_or_attn:                

            mdl.register_forward_hook(get_save_inputs_hook(name))

#momentum_dict = {}

#optim_states = {}
#optim_states["momentum"] = {}
#optim_states["grad_t-1"] = {}
#optim_states["learning_rates"] = {}

def perturbe(model, eps, scale, seed):

    seed_gen = np.random.default_rng(seed)

    with torch.no_grad():

        for name, mdl in model.named_modules():
            mlp_or_attn = ("self_attn" in name or "mlp" in name) and "_proj" in name
            emb_or_head = "embed_tokens" in name# or "lm_head" in name
            
            
                
            #if ((is_weight := hasattr(mdl, "weight")) or (is_bias := hasattr(mdl, "bias"))) and not emb_or_head:                
            if (is_weight := hasattr(mdl, "weight")) and not emb_or_head and mlp_or_attn:                
                
                #if is_bias:
                #    if mdl.bias is None:
                #        mdl.bias = torch.nn.Parameter(torch.zeros((mdl.out_features)))
            
                seed_i = seed_gen.integers(10, 1000000)
                noise = get_noise(mdl.weight.shape, seed_i)

                noise = torch.Tensor(noise).to(mdl.weight.device) * eps * scale
                #print(np.linalg.norm(mdl.weight), mdl.weight.mean(), mdl.weight.std())

                mdl.weight += noise 
                #torch.add(mdl.weight, noise, out=mdl.weight)

def update(model, grad, lr, seed):

    seed_gen = np.random.default_rng(seed)

    with torch.no_grad():

        for name, mdl in model.named_modules():
            mlp_or_attn = ("self_attn" in name or "mlp" in name) and "_proj" in name
            emb_or_head = "embed_tokens" in name# or "lm_head" in name

            
            #if ((is_weight := hasattr(mdl, "weight")) or (is_bias := hasattr(mdl, "bias"))) and not emb_or_head:                
            if (is_weight := hasattr(mdl, "weight")) and not emb_or_head and mlp_or_attn:                
                seed_i = seed_gen.integers(10, 1000000)
                noise = get_noise(mdl.weight.shape, seed_i)
                noise = torch.Tensor(noise).to(mdl.weight.device)

                update = (grad * noise)
                #if (norm := np.linalg.norm(update)) > 1:
                #    update /= norm
                #print(np.linalg.norm(update))
            

                """grad_tm1_dict = optim_states["grad_t-1"]
                learning_rates = optim_states["learning_rates"]

                if name not in grad_tm1_dict:
                    grad_tm1_dict[name] = torch.zeros_like(update).to(mdl.weight.device)
                
                if name not in learning_rates:
                    learning_rates[name] = lr * torch.ones(()).to(mdl.weight.device)
                print(name, learning_rates[name])

                gu = update * (-grad_tm1_dict[name])
                gu = gu.sum()
                #learning_rates[name] = learning_rates[name] - lr * gu
                learning_rates[name] = learning_rates[name] * (1 - lr * (gu / (1e-5+torch.linalg.norm(grad_tm1_dict[name])*torch.linalg.norm(update))))
                
                grad_tm1_dict[name] = update"""

                #momentum_dict = optim_states["momentum"]
                
                #if name not in momentum_dict:
                #    momentum_dict[name] = torch.zeros_like(update).to(mdl.weight.device)

                #momentum_dict[name] = momentum_dict[name] * 0.9 + update    

                #print(name, torch.linalg.norm(update), torch.linalg.norm(momentum_dict[name]), momentum_dict[name].mean(), momentum_dict[name].std())
                #print(update.mean(), update.std())
                #update = update / (1e-5 + update.std())
                mdl.weight -= lr * update# + mdl.weight * 0.01)
                #mdl.weight -= learning_rates[name] * (update)# + mdl.weight * 0.01)
                #mdl.weight -= lr * (momentum_dict[name])# + mdl.weight * 0.1)
                #torch.add(mdl.weight, -lr*update, out=mdl.weight)

def get_weight_scores(model, grad, seed):
    
    seed_gen = np.random.default_rng(SEED)

    scores = {}

    with torch.no_grad():

        for name, mdl in model.named_modules():
            mlp_or_attn = ("self_attn" in name or "mlp" in name) and "_proj" in name
            emb_or_head = "embed_tokens" in name or "lm_head" in name

            #if ((is_weight := hasattr(mdl, "weight")) or (is_bias := hasattr(mdl, "bias"))) and not emb_or_head:                
            if (is_weight := hasattr(mdl, "weight")) and not emb_or_head and mlp_or_attn:                
                seed_i = seed_gen.integers(10, 1000000)
                noise = get_noise(mdl.weight.shape, seed_i)
                noise = torch.Tensor(noise).to(mdl.weight.device)

                X = stored_inputs[name].to(mdl.weight.device)
                W = mdl.weight.to(mdl.weight.device)
                g = (grad * noise).to(mdl.weight.device)

                #print(name, X.shape, W.shape, g.shape)
                s = torch.einsum("bti,di,di->btdi", X, W, g).mean([0,1])
                
                scores[name] = torch.abs(s)

    return scores

def add_positive_noise(model, eps, seed):
    perturbe(model, eps, 1, seed)

def remove_positive_and_add_negative_noise(model, eps, seed):
    perturbe(model, eps, -2, seed)

def reset_noise(model, eps, seed):
    perturbe(model, eps, 1, seed)


B = 1
S = 64

SEED = 0
EPS = 1e-1 * (1/(B**0.5))
LR = 1e-2

dataset = load_dataset("wikimedia/wikipedia", "20231101.de", streaming=True)["train"]
#shuffled_dataset = ds.shuffle(seed=42, buffer_size=1000)    

def batchgen(B, S):

    curr = torch.empty((1, 1))
    batch = []

    while True:
        if curr.shape[0] < B:
            _curr = tokenizer(next(iter(dataset))["text"], return_tensors="pt")["input_ids"][0]


            num_seq = _curr.shape[-1] // S

            curr = _curr[:S*num_seq].reshape((-1, S))

        else:
            yield curr[:B, :]
            curr = curr[B:, :]
            

bg = batchgen(B, S+1)


with torch.no_grad():
    
    steps = 0
    maxsteps = 1000

    while True:

        SEED = SEED + 1
        batch = next(bg)
        
        inputs = batch[:, :-1].to(model.model.embed_tokens.weight.device)
        target = torch.nn.functional.one_hot(batch[:, 1:], config.vocab_size).type(torch.float32).to("cpu")



        #y0 = model.forward(**batch)#, attention_mask=batch["attention_mask"])

        add_positive_noise(model, EPS, SEED)        
        yp = model.forward(inputs)["logits"].to("cpu")
        #print(((y0["logits"]-yp["logits"])**2).mean())
        #print(yp["logits"].shape, target.shape)

        print(batch.shape, yp.shape, target.shape)
        lp = torch.nn.functional.cross_entropy(yp, target)

        remove_positive_and_add_negative_noise(model, EPS, SEED)
        yn = model.forward(inputs)["logits"].to("cpu")
        #print(((y0["logits"]-yn["logits"])**2).mean())
        ln = torch.nn.functional.cross_entropy(yn, target)
        
        print(lp, ln)
        projected_grad = ((lp - ln) / (2 * EPS)).item()
        print(projected_grad)



        reset_noise(model, EPS, SEED)



        
        stored_inputs_bool = True
        y0 = model.forward(inputs)["logits"].to("cpu")
        l0 = torch.nn.functional.cross_entropy(y0, target)
        print(l0)
        stored_inputs_bool = False

        weight_scores = get_weight_scores(model, projected_grad, SEED)


        mlp_groups = {}
        kv_groups = {}
        qo_groups = {}

                
        for lid in range(len(model.model.layers)):
            layer = model.model.layers[lid]

            u = layer.mlp.up_proj.weight
            g = layer.mlp.gate_proj.weight
            d = layer.mlp.down_proj.weight
            
            u_score = weight_scores[f"model.layers.{lid}.mlp.up_proj"]
            g_score = weight_scores[f"model.layers.{lid}.mlp.gate_proj"]
            d_score = weight_scores[f"model.layers.{lid}.mlp.down_proj"]

            k = layer.self_attn.k_proj.weight
            v = layer.self_attn.v_proj.weight
            
            k_score = weight_scores[f"model.layers.{lid}.self_attn.k_proj"]
            v_score = weight_scores[f"model.layers.{lid}.self_attn.v_proj"]
            
            q = layer.self_attn.q_proj.weight
            o = layer.self_attn.o_proj.weight

            q_score = weight_scores[f"model.layers.{lid}.self_attn.q_proj"]
            o_score = weight_scores[f"model.layers.{lid}.self_attn.o_proj"]


            ugd_group = torch.Tensor([max([u_score[i, :].sum(), g_score[i, :].sum(), d_score[:, i].sum()]) for i in range(u.shape[0])])
            kv_group = torch.Tensor([max([k_score[i, :].sum(), v_score[i, :].sum()]) for i in range(k.shape[0])])
            qo_group = torch.Tensor([max([q_score[i, :].sum(), o_score[:, i].sum()]) for i in range(q.shape[0])])


            
                
            cutoff_idx = torch.argsort(ugd_group, descending=True)[(ugd_group.shape[0]//100)*90]

            keep = []
            for iii in range(ugd_group.shape[0]):
                if ugd_group[iii] > ugd_group[cutoff_idx]:
                    keep.append(iii)
            keep = torch.IntTensor(keep)



            layer.mlp.up_proj.weight = torch.nn.Parameter(u[keep, :])
            layer.mlp.up_proj.in_features = u.shape[-1]
            layer.mlp.up_proj.out_features = keep.shape[-1]
            
            layer.mlp.gate_proj.weight = torch.nn.Parameter(g[keep, :])
            layer.mlp.gate_proj.in_features = g.shape[-1]
            layer.mlp.gate_proj.out_features = keep.shape[-1]
            
            layer.mlp.down_proj.weight = torch.nn.Parameter(d[:, keep])
            layer.mlp.down_proj.in_features = keep.shape[-1]
            layer.mlp.down_proj.out_features = d.shape[-1]
            
            #sorted_idx = torch.argsort(kv_group, descending=True)
            #keep = kv_group > kv_group[int(kv_group.shape[0]* 0.8)]

            #layer.self_attn.k_proj.weight = torch.nn.Parameter(k[keep, :])
            #layer.self_attn.k_proj.weight = torch.nn.Parameter(v[keep, :])

            #sorted_idx = torch.argsort(qo_group, descending=True)
            #keep = qo_group > qo_group[int(qo_group.shape[0]* 0.8)]
            
            #print(o)
            #layer.self_attn.q_proj.weight = torch.nn.Parameter(q[keep, :])
            #layer.self_attn.o_proj.weight = torch.nn.Parameter(o[:, keep])
            #print(layer.self_attn.o_proj.weight)




        y0 = model.forward(inputs)["logits"].to("cpu")
        l0 = torch.nn.functional.cross_entropy(y0, target)
        print(l0)
        exit()        
        #update(model, projected_grad, LR, SEED)


