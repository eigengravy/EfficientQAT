import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os

def update_dataset(layer, dataset, dev, attention_mask, position_ids):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                dataset.update_data(index,new_data)


def ddcl_regularization_loss(model):
    total_cost = None
    total_params = 0
    for module in model.modules():
        if isinstance(module, int_linear_fake.QuantLinear):
            cost = module.weight_quantizer.ddcl_bit_cost(module.weight)
            num_params = module.weight.numel()
            weighted_cost = cost * num_params
            total_cost = weighted_cost if total_cost is None else total_cost + weighted_cost
            total_params += num_params
    if total_cost is None:
        return None
    return total_cost / total_params


def ddcl_code_range_violation(model):
    total_rate = None
    total_params = 0
    for module in model.modules():
        if isinstance(module, int_linear_fake.QuantLinear):
            rate = module.weight_quantizer.code_range_violation(module.weight)
            num_params = module.weight.numel()
            weighted_rate = rate * num_params
            total_rate = weighted_rate if total_rate is None else total_rate + weighted_rate
            total_params += num_params
    if total_rate is None:
        return None
    return total_rate / total_params


def ddcl_rho_utilization(model):
    total = None
    groups = 0
    for module in model.modules():
        if isinstance(module, int_linear_fake.QuantLinear):
            utilization = module.weight_quantizer.rho_utilization()
            group_count = module.weight_quantizer.raw_rho.numel()
            weighted = utilization * group_count
            total = weighted if total is None else total + weighted
            groups += group_count
    if total is None:
        return None
    return total / groups

                    
def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
    wandb_run=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    
    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # step 6: start training    
    loss_func = torch.nn.MSELoss()
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        # step 6.1: replace torch.nn.Linear with QuantLinear for QAT
        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size, scheme=args.scheme)
                set_op_by_name(qlayer, name, quantlinear)  
                del module  
        qlayer.to(dev)
        
        
        # step 6.2: obtain output of full-precision model for MSE
        set_quant_state(qlayer,weight_quant=False) # deactivate quantization for obtaining ground truth
        if args.epochs > 0:
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,fp_val_inps,dev,attention_mask,position_ids)
        set_quant_state(qlayer,weight_quant=True)  # activate quantization
        
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is required for AMP training
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size 
            if args.quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if args.weight_lr > 0:
                set_weight_parameters(qlayer,True)
                param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            best_state_dict = None
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []
                ddcl_bit_cost_list = []
                ddcl_loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):    
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        reconstruction_loss = loss_func(label, quant_out)
                        ddcl_bit_cost = None
                        ddcl_loss = None
                        loss = reconstruction_loss
                        ddcl_lambda = getattr(args, "ddcl_lambda", 0.0)
                        if args.scheme == "ddcl" and ddcl_lambda > 0:
                            ddcl_bit_cost = ddcl_regularization_loss(qlayer)
                            ddcl_loss = ddcl_lambda * ddcl_bit_cost
                            loss = loss + ddcl_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    if ddcl_bit_cost is not None:
                        ddcl_bit_cost_list.append(ddcl_bit_cost.detach().cpu())
                    if ddcl_loss is not None:
                        ddcl_loss_list.append(ddcl_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    if wandb_run is not None:
                        steps_per_epoch = args.train_size // args.batch_size
                        global_step = (block_index * args.epochs + epoch) * steps_per_epoch + index
                        wandb_run.log({
                            "block_ap/reconstruction_loss": reconstruction_loss.item(),
                            "block_ap/ddcl_bit_cost": ddcl_bit_cost.item() if ddcl_bit_cost is not None else 0,
                            "block_ap/ddcl_loss": ddcl_loss.item() if ddcl_loss is not None else 0,
                            "block_ap/total_loss": loss.item(),
                            "block_ap/grad_norm": norm.item(),
                            "block_ap/block_index": block_index,
                            "block_ap/epoch": epoch,
                        }, step=global_step)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                # step 6.5: calculate validation loss
                qlayer.eval()
                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())
                qlayer.train()
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                ddcl_bit_cost_mean = torch.stack(ddcl_bit_cost_list).mean() if ddcl_bit_cost_list else torch.tensor(0.0)
                ddcl_loss_mean = torch.stack(ddcl_loss_list).mean() if ddcl_loss_list else torch.tensor(0.0)
                norm_mean = torch.stack(norm_list).mean()
                ddcl_range_violation = None
                ddcl_rho = None
                if args.scheme == "ddcl":
                    with torch.no_grad():
                        ddcl_range_violation = ddcl_code_range_violation(qlayer)
                        ddcl_rho = ddcl_rho_utilization(qlayer)
                ddcl_range_violation_mean = ddcl_range_violation.cpu() if ddcl_range_violation is not None else torch.tensor(0.0)
                ddcl_rho_mean = ddcl_rho.cpu() if ddcl_rho is not None else torch.tensor(0.0)
                current_quant_lr = quant_scheduler.get_last_lr()[0] if args.quant_lr > 0 else 0
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} ddcl_bit_cost:{ddcl_bit_cost_mean} ddcl_loss:{ddcl_loss_mean} ddcl_code_range_violation:{ddcl_range_violation_mean} ddcl_rho_utilization:{ddcl_rho_mean} quant_lr:{current_quant_lr} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")

                if wandb_run is not None:
                    steps_per_epoch = args.train_size // args.batch_size
                    epoch_step = (block_index * args.epochs + epoch + 1) * steps_per_epoch
                    epoch_metrics = {
                        "block_ap/train_loss_mean": loss_mean.item(),
                        "block_ap/val_loss_mean": val_loss_mean.item(),
                        "block_ap/ddcl_bit_cost_mean": ddcl_bit_cost_mean.item(),
                        "block_ap/ddcl_loss_mean": ddcl_loss_mean.item(),
                        "block_ap/ddcl_code_range_violation_mean": ddcl_range_violation_mean.item(),
                        "block_ap/ddcl_rho_utilization_mean": ddcl_rho_mean.item(),
                        "block_ap/grad_norm_mean": norm_mean.item(),
                        "block_ap/quant_lr": current_quant_lr,
                        "block_ap/peak_vram_mb": torch.cuda.max_memory_allocated(dev) / 1024**2,
                        "block_ap/epoch_time_s": time.time() - start_time,
                        "block_ap/block_index": block_index,
                        "block_ap/epoch": epoch,
                    }
                    if args.weight_lr > 0:
                        epoch_metrics["block_ap/weight_lr"] = weight_scheduler.get_lr()[0]
                    wandb_run.log(epoch_metrics, step=epoch_step)

                if val_loss_mean.item() < best_val_loss:
                    best_val_loss = val_loss_mean.item()
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in qlayer.state_dict().items()
                    }
                    early_stop_flag = 0
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer
            if best_state_dict is not None:
                qlayer.load_state_dict(best_state_dict)
                del best_state_dict

        # step 6.6: directly replace the weight with fake quantization
        qlayer.eval()
        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer,weight_quant=False)  # weight has been quantized inplace

        # step 6.7: update inputs of quantization model
        if args.epochs>0:
            update_dataset(qlayer,quant_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,quant_val_inps,dev,attention_mask,position_ids)
        layers[block_index] = qlayer.to("cpu")

        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round()
                zeros = zeros.clamp(module.weight_quantizer.qmin, module.weight_quantizer.qmax).cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)       
                logger.info(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
