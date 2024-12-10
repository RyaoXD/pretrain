from argparse import ArgumentParser
from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sklearn.cluster import KMeans

from models.backbones.encoder import BertEncoder, ImageEncoder
from models.basic_net import AttentionPool2d

from pdb import set_trace as st

class main_net(LightningModule):
    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)
        self.language_max_length = 112
        if self.hparams.sentence_split:
            self.sents_attn_pool = AttentionPool2d(self.language_max_length, self.text_encoder_q.embedding_dim, 64)
        else:
            self.sents_attn_pool = None

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)

        if self.hparams.symptom_prototype is None:
            self.symptom_prototype_layer = None
            self.symptom_prototype_vectors = None
        else:
            self.symptom_prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
            self.symptom_prototype_vectors = nn.Parameter(torch.rand(emb_dim, num_prototypes),
                                                          requires_grad=True)
        
        if self.hparams.class_prototype == "mapping":
            assert not self.symptom_prototype_layer is None, "No symptom_prototype_layer"
            self.class_prototype_layer = nn.Linear(num_prototypes, num_prototypes, bias=False)
            self.class_prototype_vectors = nn.Parameter(torch.rand(num_prototypes, num_prototypes),
                                                          requires_grad=True)
        else:
            self.class_prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
            self.class_prototype_vectors = nn.Parameter(torch.rand(emb_dim, num_prototypes),
                                                          requires_grad=True)
        
        self.class_proinit_pool = None
        self.symptom_proinit_pool = None
        
        if self._use_ddp_or_dpp2(self.trainer):
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn
        
    def split_sentence_merge(self, sents_feat_q, sents, sents_ids):
        n_sents,n_dim = sents_feat_q.shape
        # t = torch.zeros_like(sents_feat_q[0]).to(sents_feat_q.dtype)
        now_index = -1
        pos = 0
        batch_tokens_feat_q = torch.tensor([])
        tokens_feat_q = None
        reports_sents = []
        report_sents = []
        for sent_feat_q, sent, sent_ids in zip(sents_feat_q, sents, sents_ids):
            sent_id = sent_ids[0]
            if now_index != sent_id:
                now_index = sent_id
                if not tokens_feat_q is None:
                    batch_tokens_feat_q=torch.cat((batch_tokens_feat_q,tokens_feat_q.unsqueeze(0)),dim=0)
                    report_sents += ["[PAD]" for _ in range(self.language_max_length - len(report_sents))]
                    reports_sents.append(report_sents)
                tokens_feat_q = torch.zeros((self.language_max_length,n_dim)).to(sents_feat_q.dtype)
                report_sents = []
                pos = 0
            tokens_feat_q[pos] = sent_feat_q
            sent = ''.join([t+' ' for t in sent if (t not in ['[CLS]', '[PAD]', '[SEP]'])])
            report_sents.append(sent)
            pos += 1
        batch_tokens_feat_q=torch.cat((batch_tokens_feat_q,tokens_feat_q.unsqueeze(0)),dim=0).to(sents_feat_q.device)
        report_sents += ["[PAD]" for _ in range(self.language_max_length - len(report_sents))]
        reports_sents.append(report_sents)
        
        report_feat_q, pool_attn = self.sents_attn_pool(batch_tokens_feat_q)
        return report_feat_q, batch_tokens_feat_q, pool_attn, reports_sents
    
    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''
        
        # Forward of query image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(
            batch["imgs"])
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        if self.hparams.sentence_split:
            report_feat_q, word_feat_q, word_attn_q, sents = self.split_sentence_merge(report_feat_q,sents,batch["token_type_ids"])
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        
        if self.current_epoch == self.hparams.freeze_prototypes_epochs - 1:
            if self.hparams.class_prototype == "group":
                self.class_proinit_pool = report_emb_q if self.class_proinit_pool is None else\
                    torch.vstack((self.class_proinit_pool,report_emb_q))
            if self.hparams.symptom_prototype == "group":
                self.symptom_proinit_pool = rearrange(word_emb_q, "b nt d -> (b nt) d") if self.symptom_proinit_pool is None else\
                    torch.vstack((self.symptom_proinit_pool,rearrange(word_emb_q, "b nt d -> (b nt) d")))
        if self.current_epoch == self.hparams.freeze_prototypes_epochs:
            if not self.class_proinit_pool is None:
                
                if len(self.class_proinit_pool) < self.hparams.num_center:
                    self.class_proinit_pool = torch.vstack(self.class_proinit_pool)
                else:
                    device = self.class_proinit_pool.device
                    self.class_proinit_pool = self.class_proinit_pool.to('cpu')
                    kmean_model = KMeans(n_clusters=self.hparams.num_center,n_init='auto')
                    kmean_model.fit(self.class_proinit_pool)
                    
                    dic = {i:{"ind":[],"vector":None,"center":None} for i in range(self.hparams.num_center)}
                    for ind,n_label in enumerate(kmean_model.labels_):
                        dic[n_label]["ind"].append(ind)
                        dic[n_label]["vector"] = self.class_proinit_pool[ind].reshape(1,-1) if dic[n_label]["vector"] is None else\
                            torch.vstack((dic[n_label]["vector"],self.class_proinit_pool[ind].reshape(1,-1)))
                    for value in dic.values():
                        value["center"] = torch.mean(value["vector"],axis=0)
                    self.class_proinit_pool = torch.vstack([value["center"] for value in dic.values()]).to(device)
            if not self.symptom_proinit_pool is None:
                if len(self.symptom_proinit_pool) < self.hparams.num_center:
                    self.symptom_proinit_pool = torch.vstack(self.symptom_proinit_pool)
                else:
                    device = self.symptom_proinit_pool.device
                    self.symptom_proinit_pool = self.symptom_proinit_pool.to('cpu')
                    kmean_model = KMeans(n_clusters=self.hparams.num_center,n_init='auto')
                    kmean_model.fit(self.symptom_proinit_pool)
                    
                    dic = {i:{"ind":[],"vector":None,"center":None} for i in range(self.hparams.num_center)}
                    for ind,n_label in enumerate(kmean_model.labels_):
                        dic[n_label]["ind"].append(ind)
                        dic[n_label]["vector"] = self.symptom_proinit_pool[ind].reshape(1,-1) if dic[n_label]["vector"] is None else\
                            torch.vstack((dic[n_label]["vector"],self.symptom_proinit_pool[ind].reshape(1,-1)))
                    for value in dic.values():
                        value["center"] = torch.mean(value["vector"],axis=0)
                    self.symptom_proinit_pool = torch.vstack([value["center"] for value in dic.values()]).to(device)

        ########### Instance-level alignment ################
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_global = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        ########### Token-level alignment ################
        # cross attention patch to sentences
        if self.hparams.sentence_split:
            mask = torch.from_numpy(np.array(sents) == "[PAD]").type_as(
                batch["imgs"]).bool()
        else:
            mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
                batch["imgs"]).bool()

        if self.hparams.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                word_emb_q, patch_emb_q, patch_emb_q)
        else:
            atten_sim = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
            word_num = word_emb_q.size(1)
            # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
            atten_scores = F.softmax(
                atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
            word_atten_output = torch.bmm(atten_scores, patch_emb_q)

        word_atten_output = F.normalize(word_atten_output, dim=-1)

        word_sim = torch.bmm(
            word_emb_q, word_atten_output.permute(0, 2, 1)) / self.hparams.local_temperature

        with torch.no_grad():
            atten_weights = word_attn_q.detach()
            word_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)
            # TODO: maybe clip the tensor of 10 percentile and 90 percentile

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)

        word_sim = torch.bmm(word_emb_q, word_atten_output.permute(
            0, 2, 1)) / self.hparams.local_temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(
            word_emb_q).long().repeat(bz)
        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.

        if self.hparams.bidirectional:
            # Try not use atten layer
            if self.hparams.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)
                atten_sim[mask.unsqueeze(1).repeat(
                    1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(
                    atten_sim / self.hparams.local_temperature, dim=-1)  # bz, 196, 111
                patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            # patch_atten_output: bz, 196, 128
            if "vit" not in self.hparams.img_encoder:
                patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(
                    bz, patch_num).type_as(batch["imgs"]) / patch_num

            else:
                with torch.no_grad():
                    img_attn_map = self.img_encoder_q.model.blocks[-1].attn.attention_map.detach(
                    )
                    atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
                    patch_atten_weights = []
                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        atten_weight = atten_weight.clip(torch.quantile(
                            atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                        patch_atten_weights.append(atten_weight.clone())
                    patch_atten_weights = torch.stack(patch_atten_weights)

                patch_atten_weights /= patch_atten_weights.sum(
                    dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.hparams.local_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word

        else:

            loss_local = loss_word

        ########### Prototype-level alignment ################
        # normalize prototype layer
        with torch.no_grad():
            if self.hparams.class_prototype != "mapping":
                w = self.class_prototype_layer.weight.data.clone()
                if not self.class_proinit_pool is None and self.current_epoch == self.hparams.freeze_prototypes_epochs:
                    w[:len(self.class_proinit_pool)]=self.class_proinit_pool
                w = F.normalize(w, dim=1, p=2)
                self.class_prototype_layer.weight.copy_(w)
            if not self.symptom_prototype_layer is None:
                w = self.symptom_prototype_layer.weight.data.clone()
                if not self.symptom_proinit_pool is None and self.current_epoch == self.hparams.freeze_prototypes_epochs:
                    w[:len(self.symptom_proinit_pool)]=self.symptom_proinit_pool
                w = F.normalize(w, dim=1, p=2)
                self.symptom_prototype_layer.weight.copy_(w)

        ### Compute class prototype loss
        # Compute assign code of images
        if self.hparams.class_prototype == "mapping":
            img_proto_out = self.symptom_prototype_layer(img_emb_q)
            img_proto_out = self.class_prototype_layer(img_proto_out)
            report_proto_out = self.symptom_prototype_layer(report_emb_q)
            report_proto_out = self.class_prototype_layer(report_proto_out)
        else:
            img_proto_out = self.class_prototype_layer(img_emb_q)
            report_proto_out = self.class_prototype_layer(report_emb_q)

        # TODO: define this to hparams
        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.hparams.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.hparams.sinkhorn_iterations)         # bz, 500
            report_code = torch.exp(
                report_proto_out / self.hparams.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.hparams.sinkhorn_iterations)       # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.hparams.proto_temperature, dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.hparams.proto_temperature, dim=1)

        loss_i2t_proto = - \
            torch.mean(
                torch.sum(img_code * torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = - \
            torch.mean(
                torch.sum(report_code * torch.log(img_proto_prob), dim=1))

        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.

        ### Compute symptom prototype loss
        if not self.symptom_prototype_layer is None:
            # Compute assign code of images
            patch_proto_out = self.symptom_prototype_layer(patch_emb_q)
            word_proto_out = self.symptom_prototype_layer(word_emb_q)
            n_batch, n_patch, _ = patch_proto_out.shape
            _, n_sents, _ = word_proto_out.shape
            patch_proto_out = rearrange(patch_proto_out, "b np d -> (b np) d")
            word_proto_out = rearrange(word_proto_out, "b ns d -> (b ns) d")

            # TODO: define this to hparams
            with torch.no_grad():
                patch_code = torch.exp(
                    patch_proto_out / self.hparams.epsilon).t()
                patch_code = self.get_assignments(
                    patch_code, self.hparams.sinkhorn_iterations)         # bz, 500
                word_code = torch.exp(
                    word_proto_out / self.hparams.epsilon).t()
                word_code = self.get_assignments(
                    word_code, self.hparams.sinkhorn_iterations)       # bz, 500

            patch_proto_prob = F.softmax(
                patch_proto_out / self.hparams.proto_temperature, dim=1)
            word_proto_prob = F.softmax(
                word_proto_out / self.hparams.proto_temperature, dim=1)

            word_code = word_code.reshape((n_batch,n_sents,-1))
            word_proto_prob = word_proto_prob.reshape((n_batch,n_sents,-1))
            patch_code = patch_code.reshape((n_batch,n_patch,-1))
            patch_proto_prob = patch_proto_prob.reshape((n_batch,n_patch,-1))
            
            word_atten_weights = word_atten_weights.unsqueeze(-1).repeat(1,1,self.hparams.num_prototypes)
            patch_atten_weights = patch_atten_weights.unsqueeze(-1).repeat(1,1,self.hparams.num_prototypes)
            word_code = torch.mean(word_code*word_atten_weights,dim=1).squeeze()
            word_proto_prob = torch.mean(word_proto_prob*word_atten_weights,dim=1).squeeze()
            patch_code = torch.mean(patch_code*patch_atten_weights,dim=1).squeeze()
            patch_proto_prob = torch.mean(patch_proto_prob*patch_atten_weights,dim=1).squeeze()
            
            loss_p2w_proto = - \
                torch.mean(
                    torch.sum(patch_code * torch.log(word_proto_prob), dim=1))
            loss_w2p_proto = - \
                torch.mean(
                    torch.sum(word_code * torch.log(patch_proto_prob), dim=1))

            loss_proto += (loss_p2w_proto + loss_w2p_proto) / 2.

        ### Compute prototype orth loss
        if self.hparams.prototype_orth:
            # if not self.class_prototype_layer is None:
            #     prototype_vector = torch.squeeze(self.class_prototype_layer.weight)
            #     prototype_vector_T = torch.transpose(prototype_vector,0,1)
            #     orth_operator = torch.matmul(prototype_vector,prototype_vector_T)
            #     I_operator = torch.eye(prototype_vector.size(0),prototype_vector.size(0)).to(orth_operator.device)
            #     difference_value = orth_operator - I_operator
            #     loss_corth = (torch.norm(difference_value,p=1) ** 0.5)/difference_value.size(0)
            #     loss_orth += loss_corth
            #     # loss_orth = 100*torch.mean(torch.relu(torch.norm(difference_value,p=1) - 0))
            if not self.symptom_prototype_layer is None:
                prototype_vector = torch.squeeze(self.symptom_prototype_layer.weight)
                prototype_vector_T = torch.transpose(prototype_vector,0,1)
                orth_operator = torch.matmul(prototype_vector,prototype_vector_T)
                I_operator = torch.eye(prototype_vector.size(0),prototype_vector.size(0)).to(orth_operator.device)
                difference_value = orth_operator - I_operator
                loss_sorth = (torch.norm(difference_value,p=1) ** 0.5)/difference_value.size(0)
                loss_orth = loss_sorth
            loss_proto += self.hparams.lambda_orth * loss_orth
            
        return loss_global, loss_local, loss_proto, acc1, acc5
    
    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def training_step(self, batch, batch_idx):
        loss_global, loss_local, loss_proto, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = self.hparams.lambda_1 * loss_global + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto

        log = {
            "t_l": loss,
            "t_l_global": self.hparams.lambda_1 * loss_global,
            "t_l_local": self.hparams.lambda_2 * loss_local,
            "t_l_proto": self.hparams.lambda_3 * loss_proto,
            "t_acc1": acc1,
            "t_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    # freeze prototype layer
    def on_after_backward(self):
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            for param in self.class_prototype_layer.parameters():
                param.grad = None
            if not self.symptom_prototype_layer is None:
                for param in self.symptom_prototype_layer.parameters():
                    param.grad = None

    def validation_step(self, batch, batch_idx):
        loss_global, loss_local, loss_proto, acc1, acc5 = self(
            batch, batch_idx, "valid")

        loss = self.hparams.lambda_1 * loss_global + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto

        log = {
            "v_l": loss,
            "v_l_global": self.hparams.lambda_1 * loss_global,
            "v_l_local": self.hparams.lambda_2 * loss_local,
            "v_l_proto": self.hparams.lambda_3 * loss_proto,
            "v_acc1": acc1,
            "v_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--img_encoder", type=str,
                            default="resnet_50", help="vit_base, resnet_50")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
        # parser.add_argument("--batch_size", type=int, default=20)
        
        parser.add_argument("--sentence_split", action="store_true", default=False)
        parser.add_argument("--class_prototype", type=str,
                            default="mapping", help="random, group, mapping")
        parser.add_argument("--symptom_prototype", type=str or None,
                            default="random", help="random, group")
        parser.add_argument("--num_prototypes", type=int or None, default=500)
        parser.add_argument("--num_center", type=int, default=200)
        parser.add_argument("--prototype_orth", action="store_true", default=False)
        parser.add_argument("--lambda_orth", type=float, default=1.)
        
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        # parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--data_pct", type=float, default=1)
        return parser
    
    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs

    