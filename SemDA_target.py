import os.path as osp
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch.nn as nn
import scipy.io as io
import torch.optim as optim
from torchvision import transforms
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import  ImageList_idx, ImageList_idx_aug_fix
from sklearn.metrics import confusion_matrix
from clip.custom_clip_CA_not_contain_caption import get_coop
import clip
from src.utils.utils import *
logger = logging.getLogger(__name__)
from src.utils.crl_utils import History as history
import time
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests

# 全局缓存文本编码器，避免重复加载
LOCAL_CACHE_DIR = './local_model_cache'  # 请替换为你的实际本地路径

_TEXT_ENCODER = None



def batch_image_captioning(
    image_paths, processor, model, device, cfg, text_encoder, class_embeddings,
    batch_size=8, temperature=100.0
):
    """
    批量处理图片：生成描述、计算 CLIP 风格相似度。
    返回：
        captions_list: 每张图片的描述文本
        prob_matrix: numpy 数组，shape (num_images, num_classes)
    """
    domain = cfg.type[cfg.SETTING.S]
    classnames = cfg.classname
    classes_str = ", ".join(classnames)
    prompt = f"Describe the objects in this {domain}-image in the context of the following classes: [{classes_str}]"

    num_images = len(image_paths)
    captions_list = [""] * num_images
    prob_matrix = np.zeros((num_images, len(classnames)), dtype=np.float32)
    caption_embeddings_bank = np.zeros((num_images, 768), dtype=np.float32)
    # 分批处理
    for start_idx in tqdm(range(0, num_images, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, num_images)
        batch_paths = image_paths[start_idx:end_idx]
        batch_images = []
        valid_indices = []  # 记录批次内有效图片的原始索引

        # 读取图片（若某张失败则跳过，后续填充空）
        for i, img_path in enumerate(batch_paths):
            try:
                if img_path.startswith(('http://', 'https://')):
                    img = Image.open(requests.get(img_path, stream=True).raw).convert('RGB')
                else:
                    img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                valid_indices.append(start_idx + i)
            except Exception as e:
                print(f"读取图片失败 {img_path}: {e}")
                # 保持 captions_list 和 prob_matrix 对应位置为默认值（空字符串和零向量）

        if not batch_images:
            continue

        # 批量处理图像和文本
        inputs = processor(images=batch_images, text=[prompt] * len(batch_images), return_tensors="pt").to(device)

        # 生成描述
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
        # 解码每个生成的序列
        generated_ids = outputs.sequences  # (batch_size, seq_len)
        captions_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # 批量编码描述
        caption_embeddings = text_encoder.encode(
            captions_batch, convert_to_tensor=True, show_progress_bar=False
        ).to(device)  # (batch_size, emb_dim)

        # CLIP 风格相似度计算
        caption_norm = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)
        class_norm = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        logits_raw = caption_norm @ class_norm.T  # (batch_size, num_classes)
        logits = temperature * logits_raw

        # 存入结果
        for j, orig_idx in enumerate(valid_indices):
            captions_list[orig_idx] = captions_batch[j]
            prob_matrix[orig_idx] = logits[j].cpu().numpy()
            caption_embeddings_bank[orig_idx] = caption_embeddings[j].cpu().numpy()
    return captions_list, prob_matrix , caption_embeddings_bank





# 使用示例


def _get_text_encoder(device):
    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        _TEXT_ENCODER = SentenceTransformer(
            model_name_or_path='/home/h666/gte-base',
            device=device,
            cache_folder=LOCAL_CACHE_DIR
        )
    return _TEXT_ENCODER

model_name = "Salesforce/instructblip-flan-t5-xl"

import requests

import nltk
import numpy as np
print(nltk.data.path)
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()



def save_data_for_matlab(features, labels, save_dir='./matlab_data', filename_prefix='tsne_data'):
    """
    保存特征和标签数据，供MATLAB进行3D t-SNE可视化

    Args:
        features: (num, feature_dim) 特征张量
        labels: (num,) 标签张量
        save_dir: 保存目录
        filename_prefix: 文件名前缀
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 确保数据在CPU上并转换为numpy
    if torch.is_tensor(features):
        features_np = features.cpu().numpy()
    else:
        features_np = np.array(features)

    if torch.is_tensor(labels):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)

    # 检查维度是否匹配
    if features_np.shape[0] != labels_np.shape[0]:
        print(f"警告: 特征和标签的样本数不匹配! 特征: {features_np.shape[0]}, 标签: {labels_np.shape[0]}")

        # 截取较短的维度
        min_len = min(features_np.shape[0], labels_np.shape[0])
        features_np = features_np[:min_len]
        labels_np = labels_np[:min_len]
        print(f"已截取前 {min_len} 个样本")

    # 筛选前10个类别的数据 (0-9)
    mask = labels_np < 65
    features_filtered = features_np[mask]
    labels_filtered = labels_np[mask]

    print(f"原始数据量: {len(features_np)}")
    print(f"筛选后数据量 (0-9类): {len(features_filtered)}")


    # 方法1: 保存为 .mat 文件 (MATLAB原生格式)
    mat_filename = os.path.join(save_dir, f'{filename_prefix}.mat')
    io.savemat(mat_filename, {
        'features': features_filtered,
        'labels': labels_filtered
    })
    print(f"数据已保存为 .mat 文件: {mat_filename}")

    return {
        'features': features_filtered,
        'labels': labels_filtered,
        'save_dir': save_dir
    }
def save_for_vi(loader, netF, netB,flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netB(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        result = save_data_for_matlab(all_output, all_label, './matlab_data', 'my_new')
        print("保存成功")


        return all_label





def data_load(cfg): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

    dsets["clip"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["clip"] = DataLoader(dsets["clip"], batch_size=int(train_bs / 2), shuffle=False,
                                      num_workers=cfg.NUM_WORKERS, drop_last=False)

    dsets["source"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=2, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True)
    return dset_loaders

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
    #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
   # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    save_output = F.softmax(all_output)
    # print(predict)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    pred = all_output.argmax(dim=1).numpy()
    labels = all_label.numpy()
    cm = confusion_matrix(labels, pred)
    # 保存为 Excel
    df = pd.DataFrame(cm)
    df.to_excel('confusion_matrix.xlsx', index=False, header=False)
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, save_output
    else:
        return accuracy*100, mean_ent, save_output


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def test_time_tuning(model, inputs, optimizer, cfg, target_output, pred, caption):

    for j in range(cfg.ProDe.TTA_STEPS):
        with torch.amp.autocast('cuda'):
            output_logits,_ = model(inputs,caption)
            output = nn.Softmax(dim=1)(output_logits)
            iic_loss = IID_losses.IID_loss(output, target_output)
            ce_loss = F.cross_entropy(output, pred,reduction='none')
            ce_loss = ce_loss.mean()
            ce_loss = ce_loss.mean()
            msoftmax = output.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            loss = iic_loss + ce_loss * 5 - gentropy_loss
            loss =  ce_loss * 20
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output

def test_time_tuning_cls(model, inputs, optimizer, cfg, target_output):
    target_output = target_output.cuda()

    target_output_soft =  nn.Softmax(dim=1)(target_output)
    for j in range(cfg.ProDe.TTA_STEPS):
        with torch.amp.autocast('cuda'):
            output_logits,_ = model(inputs)

            new_target_output = target_output + output_logits
            _, index_new = torch.max(new_target_output, 1)

            new_target_output = nn.Softmax(dim=1)(new_target_output)
            output = nn.Softmax(dim=1)(output_logits)
            entropy_loss = F.cross_entropy(output, index_new)

            iic_loss = IID_losses.IID_loss(output, new_target_output) * 1.3 + entropy_loss * 0.4

        optimizer.zero_grad()
        iic_loss.backward()
        optimizer.step()
    return output

def test_time_adapt_eval_cls(input, model, optimizer, optim_state, cfg, target_output, flag=True):
    if flag==True:
        optimizer.load_state_dict(optim_state)
        output = test_time_tuning_cls(model, input, optimizer, cfg, target_output)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            model.eval()
            output,_ = model(input)
    return output

def test_time_adapt_eval(input, model, caption):

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            model.eval()
            output,_ = model(input, caption)
    return output

def train_target(cfg):



    text_inputs = clip_pre_text(cfg)
    dset_loaders = data_load(cfg)


    classnames_file = cfg.name_file

    with open(classnames_file, 'r') as f:
        classnames = [line.strip() for line in f.readlines()]

    run_BLIP = False
    if run_BLIP:
        txt_test_blip = open(cfg.test_dset_path).readlines()
        model_path = "/home/h666/insBLIP"
        text_encoder = _get_text_encoder(device)  # 你的全局单例函数

        # 加载 InstructBLIP 模型和处理器
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(device)
        processor = InstructBlipProcessor.from_pretrained(model_path, local_files_only=True)

        # 清理图片路径
        cleaned_paths = []
        for raw_path in txt_test_blip:
            if '.jpg' in raw_path:
                clean = raw_path.split('.jpg')[0] + '.jpg'
            else:
                clean = raw_path.strip().split()[0]
            cleaned_paths.append(clean)

        # ========== 预计算所有类别的文本嵌入 ==========
        classnames = cfg.classname
        print(f"预计算 {len(classnames)} 个类别的文本嵌入...")
        class_embeddings_list = []
        for cls_name in classnames:
            emb = text_encoder.encode(cls_name, convert_to_tensor=True, show_progress_bar=False).to(device)
            class_embeddings_list.append(emb)
        class_embeddings = torch.stack(class_embeddings_list)  # [num_classes, emb_dim]
        print("类别嵌入预计算完成。")



        batch_size = 16  # 根据显存调整
        captions_list, prob_matrix, caption_embeddings_bank = batch_image_captioning(
            cleaned_paths, processor, model, device, cfg, text_encoder, class_embeddings,
            batch_size=batch_size, temperature=100.0
        )


        # 保存 caption 文本
        input_path = cfg.test_dset_path
        dir_name = os.path.dirname(input_path)
        base_name = os.path.basename(input_path)
        output_base = base_name.replace('_list.txt', '_caption.txt')
        output_file = os.path.join(dir_name, output_base)
        with open(output_file, "w", encoding="utf-8") as f:
            for cap in captions_list:
                f.write(cap + "\n")
        print(f"\nCaption 已保存至 {output_file}")
        import numpy as np
        # 保存概率矩阵
        prob_matrix = np.array(prob_matrix)  # shape: (num_images, num_classes)
        matrix_save_path = os.path.join(dir_name, base_name.replace('_list.txt', '_probs_likeclip.npy'))
        np.save(matrix_save_path, prob_matrix)
        print(f"概率矩阵已保存至 {matrix_save_path}")

        text_em = np.array(caption_embeddings_bank)  # shape: (num_images, num_classes)
        text_em_path = os.path.join(dir_name, base_name.replace('_list.txt', '_text_feature.npy'))
        np.save(text_em_path, text_em)
        print(f"概率矩阵已保存至 {text_em_path}")




    input_path = cfg.test_dset_path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    matrix_save_path = os.path.join(dir_name, base_name.replace('_list.txt', '_probs_likeclip.npy'))


    input_path = cfg.test_dset_path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    text_em_path= os.path.join(dir_name, base_name.replace('_list.txt', '_text_feature.npy'))


    import numpy as np
    prob_matrix = np.load(matrix_save_path)

    text_em = np.load(text_em_path)

    prob_tensor = torch.from_numpy(prob_matrix).float().cuda()

    text_em = torch.from_numpy(text_em).float().cuda()




    model = get_coop(cfg.ProDe.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.ProDe.N_CTX, cfg.ProDe.CTX_INIT)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    ## set base network


    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()





    model.reset_classnames(cfg.classname, cfg.ProDe.ARCH)  # 老的

    param_group = []
    param_group_ib = []

    for k, v in netF.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:

            param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]

        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if cfg.OPTIM.LR_DECAY2 > 0:

            param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]

        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:

            v.requires_grad = True

            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False

    for k, v in model.prompt_learner.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]


    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    optim_state = deepcopy(optimizer_ib.state_dict())
    # model.reset_classnames(cfg, cfg.ProDe.ARCH) # 新的
    #

    Flag_reset = True
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    num_sample=len(dset_loaders["target"].dataset)


    start_test = True
    torch.cuda.reset_peak_memory_stats()
    input_path = cfg.s_dset_path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    # 去掉 '_list.txt' 后缀，得到干净的文件夹名
    folder_name = base_name.replace('_list.txt', '')  # 'clipart'

    save_weights_path = os.path.join(dir_name, folder_name)

    model.prompt_learner.load_weights(save_weights_path)

    input_path = cfg.test_dset_path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    clip_score_bank_save_path = os.path.join(dir_name, base_name.replace('_list.txt', f'_clip_{cfg.name}_score_bank.npy'))

    if os.path.isfile(clip_score_bank_save_path):
        print("文件存在")
        run_CLIP = False
    else:
        print("文件不存在")
        run_CLIP = True

    if run_CLIP == True:
        num_sample = len(dset_loaders["clip"].dataset)
        clip_score_bank = torch.randn(num_sample, cfg.class_num).cuda()
        for i in range(len(dset_loaders["clip"])):
            t=len(dset_loaders["clip"])
            try:
                inputs_test, target_batch, tar_idx = next(iter_test)
            except:
                iter_test = iter(dset_loaders["clip"])
                inputs_test, target_batch, tar_idx = next(iter_test)
            if inputs_test.size(0) == 1:
                continue

            print(i)
            inputs_test = inputs_test.cuda()


            inputs_test_clip = inputs_test
            caption = text_em[tar_idx]


            with torch.no_grad():

                clip_score = test_time_adapt_eval(inputs_test_clip, model, caption).float()
                clip_score_bank[tar_idx] = clip_score


        input_path = cfg.test_dset_path
        dir_name = os.path.dirname(input_path)
        base_name = os.path.basename(input_path)
        clip_score_bank = np.array(clip_score_bank.cpu())  # shape: (num_images, num_classes)
        clip_score_bank_save_path = os.path.join(dir_name, base_name.replace('_list.txt', f'_clip_{cfg.name}_score_bank.npy'))
        np.save(clip_score_bank_save_path, clip_score_bank)
        print(f"概率矩阵已保存至 {clip_score_bank_save_path}")

    input_path = cfg.test_dset_path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    clip_score_bank_save_path = os.path.join(dir_name, base_name.replace('_list.txt', f'_clip_{cfg.name}_score_bank.npy'))

    clip_score_bank = np.load(clip_score_bank_save_path)

    clip_score_bank = torch.from_numpy(clip_score_bank).float().cuda()

    iter_num = 0
    while iter_num < max_iter:

        try:
            (inputs_test, inputs_test_augs), target_batch, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_test_augs), target_batch, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        target_batch = target_batch.cuda()

        inputs_test = inputs_test.cuda()
        inputs_test_augs = inputs_test_augs[0].cuda() 

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        clip_score = clip_score_bank[tar_idx]
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        with torch.no_grad():
            outputs_test_new = outputs_test.clone().detach()

        if cfg.ProDe.ARCH == 'RN50':
            inputs_test_clip = inputs_test_augs
        else: 
            inputs_test_clip = inputs_test


        clip_score = clip_score.float()

        alpha = 0.5
        beta = 1

        with torch.no_grad():
            if iter_num > interval_iter * 0:
                new_clip = (outputs_test_new) + clip_score.cuda() * alpha
            else:
                new_clip = clip_score.cuda()

        BLIP_prob = prob_tensor[tar_idx, :]

        BLIP_prob_softmax = nn.Softmax(dim=1)(BLIP_prob)
        clip_score_softmax = nn.Softmax(dim=1)(clip_score)
        new_clip_softmax = nn.Softmax(dim=1)(new_clip)
        outputs_test_new_softmax = nn.Softmax(dim=1)(outputs_test_new)



        BLIP_cre_new_clip = (new_clip  + BLIP_prob * beta).argmax(dim=1)


        _,clip_index_new = torch.max(new_clip, 1)
        clip_score_sm = nn.Softmax(dim=1)(new_clip)

        iic_loss = IID_losses.IID_loss(softmax_out, outputs_test_new_softmax)
        classifier_loss = cfg.ProDe.IIC_PAR * iic_loss

        msoftmax = softmax_out.mean(dim=0)

        if  cfg.SETTING.DATASET=='office':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  cfg.SETTING.DATASET=='office-home':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  cfg.SETTING.DATASET=='VISDA-C':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 0.1 * gentropy_loss
        if cfg.SETTING.DATASET == 'domainnet126':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.ProDe.EPSILON))
            classifier_loss = classifier_loss - 0.01 * gentropy_loss
        pred = BLIP_cre_new_clip
        # pred = clip_index_new
        entropy_loss = F.cross_entropy(outputs_test, pred,reduction='none')
        # entropy_loss = (entropy_loss * weights).mean()
        entropy_loss = entropy_loss.mean()

        classifier_loss =  cfg.ProDe.GENT_PAR*entropy_loss  + classifier_loss
        #
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()



        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            netC.eval()
            label = save_for_vi(dset_loaders['test'], netF, netB, True)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list, save_output = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list

            else:
                acc_s_te, _ , save_output= cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)

            logger.info(log_str)
            netF.train()
            netB.train()
            netC.train()

        
    return netF, netB, netC


def print_cfg(cfg):
    s = "==========================================\n"    
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.ProDe.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts
