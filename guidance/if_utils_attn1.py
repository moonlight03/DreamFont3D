from transformers import logging
from diffusers import IFPipeline, DDPMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()
import gradio as gr
import numpy as np

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

global sreg, creg, sizereg, COUNT, creg_maps, sreg_maps, text_cond,reg_sizes,timesteps, atte, noattn, noresidual
sreg = 0
creg = 0
sizereg = 0
COUNT = 0
reg_sizes = []
creg_maps = []
sreg_maps = []
text_cond = []
atte = 0
noattn = 1
noresidual = 1
def preprocess_mask(mask_, h, w, device):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask
def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix
def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
    from nerf.utils import local_step
    from nerf.utils import global_step
    global atte
    if global_step >= 7000:
        atte = 100000
    t1 = local_step % 24 + atte
    t2 = t1 - 8
    ###################################################################################################3
    # if t1 != 1000000:
    if t1 >= 8 and t1 <= 13 and local_step > 24:
        global sreg, creg, COUNT, creg_maps, sreg_maps, reg_sizes, text_cond, timesteps
        COUNT += 1
        treg = torch.pow(timesteps[COUNT // 3200 // 2] / 1000, 5)  # 这个数可以调一下
        residual = hidden_states

        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)  # 2 1408 32 32 -> 2 1024 1408 ,  2 77 2816
        batch_size, sequence_length, _ = hidden_states.shape  # 2,  1024

        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)  # None

        encoder_hidden_states = text_cond
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)  # 2 77 2816


        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)  # 2 1024 1408 -> 2 1024 1408
        query = self.to_q(hidden_states)  # 2 1024 1408
        query = self.head_to_batch_dim(query)  # 44 1024 64

        key = self.add_k_proj(encoder_hidden_states)  # 2 77 2816 -> 2 77 1408
        value = self.add_v_proj(encoder_hidden_states)  # 2 77 2816 -> 2 77 1408
        key = self.head_to_batch_dim(key)  # 44 77 64
        value = self.head_to_batch_dim(value)  # 44 77 64

        # 44 1024 64  44 1101 64lx
        ###########################################################################################333

        empty1 = torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device)
        key = key.transpose(-1, -2)
        sim = torch.baddbmm(empty1, query, key, beta=0, alpha=self.scale)


        min_value = sim[int(sim.size(0) / 2):].min(-1)[0].unsqueeze(-1)  # 22 1024 1
        max_value = sim[int(sim.size(0) / 2):].max(-1)[0].unsqueeze(-1)  # 22 1024 1
        mask = creg_maps[t2][sim.size(1)].repeat(self.heads, 1, 1)  # 22 1024 77
        size_reg = reg_sizes[t2][sim.size(1)].repeat(self.heads, 1, 1)  # 22 1024 1

        sim[int(sim.size(0) / 2):] += (mask > 0) * size_reg * creg * treg * (max_value - sim[int(sim.size(0) / 2):])  # 5 4096 77
        sim[int(sim.size(0) / 2):] -= ~(mask > 0) * size_reg * creg * treg * (sim[int(sim.size(0) / 2):] - min_value)  # 5 4096 77
        attention_probs = sim.softmax(dim=-1)  # sa: 10 4096 4096  ca: 10 4096 77
        attention_probs = attention_probs.to(query.dtype)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
        ###############################################################################################

        # linear proj
        hidden_states = self.to_out[0](hidden_states)  # 2 1024 1408
        # dropout
        hidden_states = self.to_out[1](hidden_states)  # 2 1024 1408

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)  # 2 1048 32 32
        hidden_states = hidden_states * noattn + residual * noresidual

        return hidden_states
    else:
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1,2)  # 2 1408 32 32 -> 2 1024 1408 ,  2 77 2816
        batch_size, sequence_length, _ = hidden_states.shape  # 2,  1024

        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)  # None

        if encoder_hidden_states is None:  # 如果这个值是空，就是Self Attention，否则就是Cross Attention
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)  # 2 77 2816

        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)  # 2 1024 1408 -> 2 1024 1408

        query = self.to_q(hidden_states)  # 2 1024 1408
        query = self.head_to_batch_dim(query, out_dim=4)  # 2 22 1024 64

        encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)  # 2 77 2816 -> 2 77 1408
        encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)  # 2 77 2816 -> 2 77 1408
        encoder_hidden_states_key_proj = self.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)  # 2 22 77 64
        encoder_hidden_states_value_proj = self.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)  # 2 22 77 64

        if not self.only_cross_attention:  # 图像噪声当Q，text emb和图像噪声cat起来当K,V  #  在IF中总是执行这行命令
            key = self.to_k(hidden_states)  # 2 1024 1408 -> 2 1024 1408
            value = self.to_v(hidden_states)  # 2 1024 1408 -> 2 1024 1408
            key = self.head_to_batch_dim(key, out_dim=4)  # 2 1024 1408 -> 2 22 1024 64
            value = self.head_to_batch_dim(value, out_dim=4)  # 2 1024 1408 -> 2 22 1024 64
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)  # 2 22 1101 64， 77+1024=1101
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            key = encoder_hidden_states_key_proj  # 图像噪声当Q，text emb当K,V
            value = encoder_hidden_states_value_proj

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )  # 2 22 1024 64， Q,K,V计算的结果
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, residual.shape[1])  # 2 1024 1408

        # linear proj
        hidden_states = self.to_out[0](hidden_states)  # 2 1024 1408
        # dropout
        hidden_states = self.to_out[1](hidden_states)  # 2 1024 1408

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)  # 2 1048 32 32
        hidden_states = hidden_states * noattn + residual * noresidual

        return hidden_states


def process_example(layout_path, all_prompts):
    # all_prompts = all_prompts.split('***')

    binary_matrixes = []
    colors_fixed = []

    im2arr = np.array(Image.open(layout_path).convert('RGB'))[:,:,:3]
    unique, counts = np.unique(np.reshape(im2arr, (-1, 3)), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)

    binary_matrix = create_binary_matrix(im2arr, (0, 0, 0))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_ * (255, 255, 255) + (1 - binary_matrix_) * (50, 50, 50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    for i in range(len(all_prompts) - 1):
        r, g, b = unique[sorted_idx[i]]
        if any(c != 255 for c in (r, g, b)) and any(c != 0 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r, g, b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    visibilities = []
    colors = []
    prompts = []
    for n in range(12):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
        prompts.append(gr.update())

    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]
        prompts[n] = all_prompts[n + 1]

    return binary_matrixes, prompts

class IF(nn.Module):
    def __init__(self, device, vram_O, t_range=[0.02, 0.98], opt=None):
        super().__init__()

        self.device = device

        print(f'[INFO] loading DeepFloyd IF-I-XL...')

        model_key = "DeepFloyd/IF-I-XL-v1.0"

        is_torch2 = torch.__version__[0] == '2'

        # Create model
        pipe = IFPipeline.from_pretrained(model_key, variant="fp16", torch_dtype=torch.float16)
        if not is_torch2:
            pipe.enable_xformers_memory_efficient_attention()

        if vram_O:
            # pipe.unet.to(memory_format=torch.channels_last)
            # pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        self.pipe = pipe
        ###################################################################################################################3
        ###################################################################################################################3
        global reg_sizes, creg_maps, sreg_maps, atte, timesteps, text_cond,noattn, noresidual
        atte = opt.atte
        noattn = opt.noattn
        noresidual = opt.noresidual
        self.MAX_COLORS = 12
        timesteps = self.scheduler.timesteps
        self.sp_sz = self.unet.sample_size
        seed, self.creg_, self.sreg_, self.sizereg_, self.bsz = 114972190, 1, 0.3, 1, 1
        self.master_prompt = opt.text
        all_prompts = opt.atte_text.split('***')
        paths = ['{}-{}.png'.format(opt.atte_img, i) for i in [0, 45, 135, 180, 225, 315]]
        for i in range(len(paths)):
            path_t = paths[i]
            binary_matrixes, prompts = process_example(path_t, all_prompts)
            self.process_generation(binary_matrixes, self.creg_, self.sreg_, self.sizereg_, self.bsz, self.master_prompt, *prompts)
        c1 = 0
        for _module in self.unet.modules():
            if _module.__class__.__name__ == "Attention":
                c1 += 1
                _module.__class__.__call__ = mod_forward
        print(c1)

        ###################################################################################################################3

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        print(f'[INFO] loaded DeepFloyd IF-I-XL!')
    def process_generation(self, binary_matrixes, creg_, sreg_, sizereg_, bsz, master_prompt, *prompts):

        global sreg, creg, sizereg, COUNT, creg_maps, sreg_maps, text_cond, reg_sizes
        with torch.no_grad():
            creg, sreg, sizereg = creg_, sreg_, sizereg_
            clipped_prompts = prompts[:len(binary_matrixes)]
            prompts = [master_prompt] + list(clipped_prompts)  # 一个主五个副
            layouts = torch.cat([preprocess_mask(mask_, self.sp_sz, self.sp_sz, self.device) for mask_ in binary_matrixes])  # 5 1 64 64

            prompt1 = self.pipe._text_preprocessing(prompts, clean_caption=False)
            text_input = self.tokenizer(prompt1, padding='max_length', max_length=77, truncation=True, return_length=True, add_special_tokens=True, return_tensors='pt')
            cond_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]


            prompt2 = self.pipe._text_preprocessing([""], clean_caption=False)
            inputs2 = self.tokenizer(prompt2, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
            uncond_embeddings = self.text_encoder(inputs2.input_ids.to(self.device))[0]

            ###########################
            ###### prep for sreg ######
            ###########################

            sreg_maps_ = {}  # 1 ? ?
            reg_sizes_ = {}  # 不同形式的layouts 1 ? 1

            for r in range(4):
                res = int(self.sp_sz / np.power(2, r))
                layouts_s = F.interpolate(layouts, (res, res), mode='nearest')
                a = layouts_s.view(layouts_s.size(0), 1, -1)
                b = layouts_s.view(layouts_s.size(0), -1, 1)
                c = a * b
                layouts_s = c.sum(0).unsqueeze(0).repeat(bsz, 1, 1)  # 进行叠加之后就变成全Mask了
                reg_sizes_[np.power(res, 2)] = 1 - sizereg * layouts_s.sum(-1, keepdim=True) / (np.power(res, 2))
                sreg_maps_[np.power(res, 2)] = layouts_s

            ###########################
            ###### prep for creg ######
            ###########################
            pww_maps = torch.zeros(1, 77, 64, 64).to(self.device)
            for i in range(1, len(prompts)):
                wlen = text_input['length'][i] - 1
                widx = text_input['input_ids'][i][: wlen]
                for j in range(77):  # 77意味着这句话的长度是77个单词
                    try:
                        if (text_input['input_ids'][0][j:j + wlen] == widx).sum() == wlen:
                            pww_maps[:, j:j + wlen, :, :] = layouts[i - 1:i]  # 新建一个空的emb嵌入，在对应的单词index位置上，嵌入不同的layout emb
                            cond_embeddings[0][j:j + wlen] = cond_embeddings[i][1:1 + wlen]  # 主提示词的对应单词index位置上，嵌入了分量提示词的emb
                            break
                    except:
                        raise gr.Error("Please check whether every segment prompt is included in the full text !")
                        return

            creg_maps_ = {}
            for r in range(4):
                res = int(self.sp_sz / np.power(2, r))
                layout_c = F.interpolate(pww_maps, (res, res), mode='nearest').view(1, 77, -1).permute(0, 2, 1).repeat(bsz,1, 1)
                creg_maps_[np.power(res, 2)] = layout_c

            text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz, 1, 1)])  # 2 77 4096
            encoder = self.unet.encoder_hid_proj.to(self.device)
            text_cond = encoder(text_cond)
            creg_maps.append(creg_maps_)
            sreg_maps.append(sreg_maps_)
            reg_sizes.append(reg_sizes_)
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        # TODO: should I add the preprocessing at https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#LL486C10-L486C28
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, grad_scale=1):

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample  # text_embeddings: 2 77 4096
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # TODO: how to use the variance here?
            # noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]

        return loss

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, grad_scale=1):

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts        

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * (1 + K))
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)



        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]

        return loss

    @torch.no_grad()
    def produce_imgs(self, text_embeddings, height=64, width=64, num_inference_steps=50, guidance_scale=7.5):

        images = torch.randn((1, 3, height, width), device=text_embeddings.device, dtype=text_embeddings.dtype)
        images = images * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            model_input = torch.cat([images] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)

            # predict the noise residual
            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        images = (images + 1) / 2

        return images


    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img
        imgs = self.produce_imgs(text_embeds, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=64)
    parser.add_argument('-W', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = IF(device, opt.vram_O)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




