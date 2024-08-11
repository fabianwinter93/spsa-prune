from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import torch
import numpy as np

import os

from scipy.stats.qmc import Sobol, discrepancy
from scipy.stats import norm, truncnorm

#REPO_ID = "mayflowergmbh/Wiedervereinigung-7b-dpo-laser"
REPO_ID = "VAGOsolutions/SauerkrautLM-1.5b"


config = AutoConfig.from_pretrained(REPO_ID)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

#config.num_hidden_layers = 1
#config.hidden_size = 256
#config.intermediate_size = 896
#config.num_attention_heads = 8

#model = AutoModelForCausalLM.from_config(config)
model = AutoModelForCausalLM.from_pretrained(REPO_ID, device_map="auto", config=config, local_files_only=True)

"""
with torch.no_grad():
    for name, mdl in model.named_modules():
        mlp_or_attn = ("self_attn" in name or "mlp" in name) and "_proj" in name
        emb_or_head = "embed_tokens" in name or "lm_head" in name

        if mlp_or_attn or emb_or_head:

            u, s, v = torch.linalg.svd(mdl.weight.data)
            k = torch.linalg.matrix_rank(mdl.weight.data)

            s = s[:k]
            
            s = torch.diag(s[:k])
            mdl.weight.data = (u[:, :k] @ s) @ v[:k, :]"""
            
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


#momentum_dict = {}

optim_states = {}
optim_states["momentum"] = {}
optim_states["grad_t-1"] = {}
optim_states["learning_rates"] = {}

def perturbe(model, scale, sign, seed):

    seed_gen = np.random.default_rng(seed)

    with torch.no_grad():

        for name, mdl in model.named_modules():
            #mlp_or_attn = ("self_attn" in name or "mlp" in name) and "_proj" in name
            emb_or_head = "embed_tokens" in name or "lm_head" in name
            

            if hasattr(mdl, "weight") and not emb_or_head:                
                seed_i = seed_gen.integers(1000, 100000)
                noise = get_noise(mdl.weight.shape, seed_i)

                noise = torch.Tensor(noise).to(mdl.weight.device) * scale * sign
                #print(np.linalg.norm(mdl.weight), mdl.weight.mean(), mdl.weight.std())

                mdl.weight += noise 


def update(model, grad, lr, seed):

    seed_gen = np.random.default_rng(seed)

    with torch.no_grad():

        for name, mdl in model.named_modules():
            #mlp_or_attn = ("self_attn" in name or "mlp" in name) and "_proj" in name
            emb_or_head = "embed_tokens" in name or "lm_head" in name

            
            if hasattr(mdl, "weight") and not emb_or_head:
                    seed_i = seed_gen.integers(1000, 100000)
                    noise = get_noise(mdl.weight.shape, seed_i)
                    
                    noise = torch.Tensor(noise).to(mdl.weight.device)
                    update = (grad * noise)
                    #if (norm := np.linalg.norm(update)) > 1:
                    #    update /= norm
                    #print(np.linalg.norm(update))
                

                    grad_tm1_dict = optim_states["grad_t-1"]
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
                    
                    grad_tm1_dict[name] = update

                    #momentum_dict = optim_states["momentum"]
                    
                    #if name not in momentum_dict:
                    #    momentum_dict[name] = torch.zeros_like(update).to(mdl.weight.device)

                    #momentum_dict[name] = momentum_dict[name] * 0.9 + update    

                    mdl.weight -= learning_rates[name] * (update)# + mdl.weight * 0.01)
                    #mdl.weight -= lr * (momentum_dict[name])# + mdl.weight * 0.1)

def add_positive_noise(model, scale, seed):
    perturbe(model, scale, 1, seed)

def remove_positive_and_add_negative_noise(model, scale, seed):
    perturbe(model, scale, -2, seed)

def reset_noise(model, scale, seed):
    perturbe(model, scale, 1, seed)


SEED = 1000
SCALE = 1e-0
LR = 1e-1

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
            

bg = batchgen(8, 128+1)


with torch.no_grad():
    
    steps = 0
    maxsteps = 1000

    while True:

        #SCALE = np.random.uniform(0, 1)

        SEED = SEED + 1
        #texts = [next(iter(dataset))["text"] for _ in range(8)]
        #batch = tokenizer.batch_encode_plus(texts, return_tensors="pt")
        batch = next(bg)
        
        inputs = batch[..., :-1].to(model.model.embed_tokens.weight.device)
        target = torch.nn.functional.one_hot(batch[..., 1:], config.vocab_size).type(torch.float32).to("cpu")



        #y0 = model.forward(**batch)#, attention_mask=batch["attention_mask"])

        add_positive_noise(model, SCALE, SEED)        
        yp = model.forward(inputs)["logits"].to("cpu")
        #print(((y0["logits"]-yp["logits"])**2).mean())
        #print(yp["logits"].shape, target.shape)

        print(batch.shape, yp.shape, target.shape)
        lp = torch.nn.functional.cross_entropy(yp, target)

        remove_positive_and_add_negative_noise(model, SCALE, SEED)
        yn = model.forward(inputs)["logits"].to("cpu")
        #print(((y0["logits"]-yn["logits"])**2).mean())
        ln = torch.nn.functional.cross_entropy(yn, target)
        
        print(lp, ln)
        projected_grad = ((lp - ln) / (2 * SCALE)).item()
        print(projected_grad)



        reset_noise(model, SCALE, SEED)
        #y00 = model.forward(**batch)
        #print(((y0["logits"]-y00["logits"])**2).mean())
        #l0 = torch.nn.functional.cross_entropy(y00["logits"], target)
        
        update(model, projected_grad, LR, SEED)

        #yu = model.forward(**batch)
        #print(((y0["logits"]-yu["logits"])**2).mean())
        #lu = torch.nn.functional.cross_entropy(yu["logits"], target)
        
