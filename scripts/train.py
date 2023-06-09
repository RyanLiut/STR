from __future__ import print_function
from enum import Flag
import random
import numpy as np
from sklearn import cluster
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
from tqdm import tqdm
import pickle
import argparse
import sys
import os
import json
import pandas as pd
import time
sys.path.insert(0,'./modules')
sys.path.insert(0,'./')
from modules.data_loader import COCO_Dataset, global_shuffle, my_collate
from modules.textmodules import TextDecoder, GaussPrior, PostFusion, TextEncoderMask, TextEncoderObj
from modules.utils import sequence_cross_entropy_with_logits, AsymmetricLoss
import inflect
from eval import eval
from eval import append2csv
from tensorboardX import SummaryWriter
import logging
logger = logging.getLogger(__name__)
from dotmap import DotMap
import h5py
inflect = inflect.engine()

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

def get_properseq(bottom_up_features, bottom_up_classes, seq, mask_seq, obj_seq, seq_lengths):
	sorted_bottom_up_features = torch.zeros(bottom_up_features.size())
	if bottom_up_classes is not None:
		sorted_bottom_up_classes = torch.zeros(bottom_up_classes.size())
	else:
		sorted_bottom_up_classes = None
	sort_order = np.argsort(seq_lengths)[::-1]
	seq_sorted = np.zeros((len(seq),maxlength))
	mask_seq_sorted = np.zeros((len(seq),maxlength))
	if obj_seq is not None:
		obj_seq_sorted = np.zeros((len(seq),maxlength))
	else:
		obj_seq_sorted = None
	for idx in range(len(seq)):
		a = sort_order[idx]
		seq_sorted[idx,0:seq_lengths[a]] = seq[a,:seq_lengths[a]]
		mask_seq_sorted[idx,0:seq_lengths[a]] = mask_seq[a,:seq_lengths[a]]
		if obj_seq is not None:
			obj_seq_sorted[idx,0:seq_lengths[a]] = obj_seq[a,:seq_lengths[a]]
		sorted_bottom_up_features[idx] = bottom_up_features[a]
		if bottom_up_classes is not None:
			sorted_bottom_up_classes[idx] = bottom_up_classes[a]

	seq_sorted = seq_sorted.astype(np.float64)
	mask_seq_sorted = mask_seq_sorted.astype(np.float64)
	if obj_seq is not None:
		obj_seq_sorted = obj_seq_sorted.astype(np.float64)
	seq_lengths = seq_lengths[sort_order].astype(np.int32)
	seq_sorted = torch.LongTensor(seq_sorted.astype(np.float64)).to(device)
	mask_seq_sorted = torch.LongTensor(mask_seq_sorted.astype(np.float64)).to(device)
	if obj_seq is not None:
		obj_seq_sorted = torch.LongTensor(obj_seq_sorted.astype(np.float64)).to(device)
	seq_lengths = torch.LongTensor(seq_lengths.astype(np.int32)).to(device) 
	return sorted_bottom_up_features,sorted_bottom_up_classes,seq_sorted,mask_seq_sorted,obj_seq_sorted, seq_lengths, sort_order


def get_loss( logits, targets, target_mask, pos_loss=True):
	target_lengths = torch.sum(target_mask, dim=-1).float()
	return target_lengths * sequence_cross_entropy_with_logits(
		logits, targets, target_mask, average=None, pos_loss=pos_loss
	)	

def get_SC_loss( logits, targets_sent, im_idx, cidx, gamma_neg):
	asLoss = AsymmetricLoss(gamma_neg=gamma_neg)
	targets = torch.zeros_like(logits)
	# modify if necessary to avoid two-loop.
	'''
	# Exp1: Just for one topic sentence
	for i in range(targets_sent.shape[0]):
		for j in range(targets_sent.shape[1]):
			if targets_sent[i,j].item() in word2SC:
				targets[i][word2SC[targets_sent[i,j].item()]] = 1
	'''

	# Exp2: for sentences in a cluster
	for ix, (im_id, c_id) in enumerate(zip(im_idx, cidx)):
		targets[ix][SCix_byim[str(im_id)][str(c_id)]] = 1
	
	return asLoss(logits, targets)

def get_CL_loss(logits, targets, target_mask, seq_len_per_video, n_shuffle=1,mode="Train", cluster_ids=None, im_idx=None):
	'''
	To get a loss with contrastive learning
	And make the caption set w.r.t the same video diverse as much as possible.
	'''
	pos_loss = get_loss(logits, targets, target_mask)
	pred_targets = logits.argmax(dim=-1).contiguous()

	ix_byImCid = {}
	for iix, (iid, cid) in enumerate(zip(im_idx, cluster_ids)):
		if iid not in ix_byImCid:
			ix_byImCid[iid] = {}

		if cid not in ix_byImCid[iid]:
			ix_byImCid[iid][cid] = []
		ix_byImCid[iid][cid].append(iix)

	if mode == "Val":
		return torch.mean(pos_loss), torch.tensor(.0).cuda()

	def get_psedo_loss(pred_targets):
		all_shuff_ix = []
		shuff_num = 0
		ori_ix = list(range(sum(seq_len_per_video)))
		hist_item_ix = {}
		for ix,_ in enumerate(seq_len_per_video):
			hist_item_ix[ix] = []
		all_neg_loss = 0.
		for _ in range(n_shuffle):
			'''
			shuff_ix = []
			for ix,i in enumerate(seq_len_per_video):
				s = 0
				if ix > 0:
					s = sum(seq_len_per_video[:ix])
				item_ix = list(range(s,s+i))
				while item_ix in hist_item_ix[ix] or sum([ii==jj for ii,jj in zip(ori_ix[s:s+i],item_ix)]) > 0:
					random.shuffle(item_ix)
				shuff_ix += item_ix
				hist_item_ix[ix].append(item_ix)
			'''
			for k,v in ix_byImCid.items():
				for kk,vv in v.items():
					np.random.shuffle(vv)
			
			shuff_ix=[vvv for k,v in ix_byImCid.items() for kk,vv in v.items() for vvv in vv]
			psedo_targets = pred_targets[shuff_ix]

			psedo_target_mask = torch.zeros(target_mask.shape).bool().cuda()
			for i in range(psedo_targets.shape[0]):
				for j in range(psedo_targets.shape[1]):
					psedo_target_mask[i,j] = True
					if psedo_targets[i,j] == 2:
						psedo_target_mask[i,j] = j + 1
						break

			all_neg_loss += get_loss(logits, psedo_targets, psedo_target_mask, pos_loss=False)#code change! to make the loss positive

		return all_neg_loss / n_shuffle

	avg_pos_loss = torch.mean(pos_loss)
	avg_neg_loss = torch.mean(get_psedo_loss(pred_targets))
	# avg_neg_loss = 0.0
	if avg_neg_loss.isnan().item() or avg_pos_loss.isnan().item():
		print("Stop here.")
		# debug = get_loss(logits, targets, target_mask)

	return avg_pos_loss, avg_neg_loss

def DiffDistsLoss(mask_q_means, mask_q_logs, obj_q_means, obj_q_logs, cluster_ids, im_idx, seq_len):
	'''
	Exp: Let different distributions within the same clusters diverge more.
	'''
	bs = mask_q_means.shape[0]
	loss = torch.zeros(bs).cuda()
	ix_byImCid = {}
	for iix, (iid, cid) in enumerate(zip(im_idx, cluster_ids)):
		if iid not in ix_byImCid:
			ix_byImCid[iid] = {}

		if cid not in ix_byImCid[iid]:
			ix_byImCid[iid][cid] = []
		ix_byImCid[iid][cid].append(iix)
		
	for i in range(bs):
		anchor_mask_means, anchor_mask_logs = mask_q_means[i], mask_q_logs[i]
		anchor_obj_means, anchor_obj_logs = obj_q_means[i], obj_q_logs[i]
		neg_ix = ix_byImCid[im_idx[i]][cluster_ids[i]]
		neg_mask_means, neg_mask_logs = mask_q_means[neg_ix], mask_q_logs[neg_ix]
		neg_obj_means, neg_obj_logs = obj_q_means[neg_ix], obj_q_logs[neg_ix] 

		kl_div_mask = 0.5*(anchor_mask_logs - neg_mask_logs) + (neg_mask_logs.exp() +(neg_mask_means - anchor_mask_means)**2 )/(2*anchor_mask_logs.exp()) - 0.5
		kl_div_mask = torch.sum(kl_div_mask[:,:seq_len[i],:])

		kl_div_obj = 0.5*(anchor_obj_logs - neg_obj_logs) + (neg_obj_logs.exp() +(neg_obj_means - anchor_obj_means)**2 )/(2*anchor_obj_logs.exp()) - 0.5
		kl_div_obj = torch.sum(kl_div_obj[:,:seq_len[i],:])

		loss_per_item = kl_div_obj + kl_div_mask
		loss[i] = loss_per_item / (seq_len[i] * len(neg_ix))

	return loss

def fill_slot_with_class(mask_seq, alphas, bottom_up_classes):
	
	for j in range(1,mask_seq.size(1)):
		slot_idxs = torch.nonzero(mask_seq[:,j] == 4)[:,0].cpu().numpy().tolist()
		slot_idxs_p = torch.nonzero(mask_seq[:,j] == 3)[:,0].cpu().numpy().tolist()

		
		if len(slot_idxs) > 0:
			max_attn_idx = torch.argmax(alphas[slot_idxs,j-1,:],dim=1).cpu().numpy().tolist()
			max_attn_class = bottom_up_classes[slot_idxs,max_attn_idx]
			mask_seq[slot_idxs,j] = max_attn_class.long().cuda()

		if len(slot_idxs_p) > 0:
			max_attn_idx = torch.argmax(alphas[slot_idxs_p,j-1,:],dim=1).cpu().numpy().tolist()
			max_attn_class = bottom_up_classes[slot_idxs_p, [idx + 100 for idx in max_attn_idx]]
			mask_seq[slot_idxs_p,j] = max_attn_class.long().cuda()	
	return mask_seq

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

global_result_path = "datasets/gresult.csv"

def append2csv(config, logger, best_all_score_acc, best_all_score_div, best_epoch=None):
	# Automatically save the results to csv
	df = pd.DataFrame() # change it later to append
	df['dataset'] = config['dname']
	df['folder_name'] = config['res_dir']
	df['best_epoch'] = best_epoch
	df['#sample(B_S)'] = str(config['val_beam_width'])+'_'+str(config['val_num_samples'])
	df['B4'], df['C'], df['R'], df['M'], df['S'] = round(best_all_score_acc['oracle_Bleu_4'],4), \
	    round(best_all_score_acc['oracle_CIDEr'],4), round(best_all_score_acc['oracle_ROUGE_L'],4),\
		round(best_all_score_acc['oracle_METEOR'],4), None
	df['pred_file_name'] = "best_val_pred_re.json"
	df['#samples'] = "Rerank5 [LEN]"
	df['Div1'], df['Div2'], df['gDiv1'], df['lUnis'], df['gUnis'], df['mB1'], df['mB2'], df['mB3'], df['mB4'] = \
		round(best_all_score_div['Div1'],4), round(best_all_score_div['Div2'],4), round(best_all_score_div['gDiv1'],4),\
		round(best_all_score_div['Unisent'],4), round(best_all_score_div['gUni'],4), round(best_all_score_div['mBLeu_1'],4), \
		round(best_all_score_div['mBLeu_2'],4), round(best_all_score_div['mBLeu_3'],4), round(best_all_score_div['mBLeu_4'],4)

	if os.path.exists(global_result_path):
		df_g = pd.read_csv(global_result_path)
		df_g.append(df, ignore_index=True)
		logger.debug("The result path has been created to %s!"%global_result_path)
	else:
		df_g = df
		logger.debug("The result path has been updated %s!"%global_result_path)
	df_g.to_csv(global_result_path, index=False)

if __name__== "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='msvd', help='Experiment settings.')
	parser.add_argument('--lr_byepoch', action="store_true", help="Whether to update the lr by epoch.")
	parser.add_argument('--use_one_cap', action="store_true", help="Whether only to use one caption for one visual example.")
	parser.add_argument('--use_one_cap_rand', action="store_true", help="whether to randomly pick candidate caption when use_one_cap")
	parser.add_argument('--use_z', action="store_true", help="Whether to use latent variable z")
	parser.add_argument('--max_es_cnt', default=10, type=int, help='The max early stop counts.')
	parser.add_argument('--res_dir_root', default='./results', type=str, help='The root of pred file path.')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--val_num_samples', default=10, type=int)
	parser.add_argument('--val_beam_width', default=1, type=int)
	parser.add_argument('--val_group_size', default=1, type=int)
	parser.add_argument('--dl', default=0.2, type=float)
	parser.add_argument('--resume_from', type=str, default=None, help="the directory which the model resumes from.")
	parser.add_argument('--max_epoch', type=int, default=None, help="When resuming, the new larger iteration.")
	parser.add_argument('--z_indeps', action="store_true", help="whether z2 is independent on z1.")
	parser.add_argument('--seq_CVAE', action="store_true", help="Whether uses seq_CVAE.")
	parser.add_argument('--learning_rate', default=0.015, type=float, help="The initialized lr.")
	parser.add_argument('--kl_rate', default=1.0, type=float, help="The kl rate.")
	parser.add_argument('--save_pos', action="store_true", help="Whether to save posteriori info")
	parser.add_argument('--use_pos', default=None, help="Whether to use posteriori. (Path)")
	parser.add_argument('--nval', default=None, help="the sample number in the validation set")
	parser.add_argument('--val_coco_format', default=None, help='the val coco file')
	parser.add_argument('--val_psedo_cocoformat', default=None, help='the val coco file')
	parser.add_argument('--use_CL_loss', action="store_true", help='Whether to use CL_loss')
	parser.add_argument('--cl_loss_prop', default=1.0, type=float, help='Propotion of negative CL_loss when use_CL_loss')
	parser.add_argument('--div_topK', default=5, help="TopK captions used to evaluate the diversity.")
	parser.add_argument('--coco_class', default=None, help="verb or noun path for ATVAE")
	parser.add_argument('--mark', default='', help="Some additional info adding to the file name.")
	parser.add_argument('--visualize', action="store_true", help="To visualize the weights of different objects.")
	parser.add_argument('--scale_o', default=0.35, type=float, help="The scale of object variance.")
	parser.add_argument('--non_gshuffle', action="store_true", help="Whether shuffle (default: false).")
	parser.add_argument('--isphase2', action="store_true", help="whether in phrase 2")
	parser.add_argument('--batch_size', default=None, help="batch size for both train and test")
	parser.add_argument('--latent_dim_tx', default=None, help="half of the latent dimension")
	parser.add_argument('--rand_seed', default=1021, type=int)
	parser.add_argument('--nclusters', default=5, type=int)
	parser.add_argument('--pathToData', default=None, type=str)
	parser.add_argument('--best_metrics', default=None, type=str)
	parser.add_argument('--save_all_ckpts', action='store_true', help='whether to save all the ckpts, warning about MEM!')
	parser.add_argument('--use_SC_loss', action="store_true", help='whether to use SC prediction loss')
	parser.add_argument('--sc_rate', type=float, default=0.1, help='Prediction loss weight')
	parser.add_argument('--sc_gamma_neg', type=float, default=4, help="gammar negative for SC loss")
	parser.add_argument('--SC_latent_dim', default=256, help="latent dimension for SC classification")
	parser.add_argument('--pos_div_path', default=None, help="the path for diversity evaluation.")
	parser.add_argument('--use_diff_dists_loss', action="store_true", help='Whether to use different distributions')
	parser.add_argument('--di_rate', default=0.1, type=float, help="weight for different distributions loss")
	
	
	args = parser.parse_args()

	if not args.resume_from is None:
		logger.info("Resume training from %s" % args.resume_from)
		ckpt_path = args.resume_from+'/best_model.ckpt'
		assert os.path.exists(ckpt_path), "There should be a best model in %s" % args.resume_from
		logger.info("Load the ckpt file from %s" % ckpt_path)
		with open(ckpt_path, "rb") as f:
			best_ckpt = torch.load(f)
		config = best_ckpt['model_config']
		dname = config['dname']
		if not args.max_epoch is None:
			config['max_epoch'] = int(args.max_epoch)
		logfile = "debug_re4.log"
		config['save_pos'] = args.save_pos
		config['use_pos'] = args.use_pos
		config['resume_from'] = args.resume_from
		# config['val_num_samples'] = 2 #DELETE LATER!!
		# logger.warning("Delete later!!!")
		config['pathToData'] = args.pathToData
		config['gshuffle'] = bool(1 - args.non_gshuffle)
		config['res_dir'] = args.resume_from
		config['kl_rate'] = args.kl_rate
		# config['rand_seed'] = args.rand_seed
		# config['use_SC_loss'] = args.use_SC_loss
		# config['use_CL_loss'] = args.use_CL_loss
		# config['cl_loss_prop'] = args.cl_loss_prop
		# config['learning_rate'] = args.learning_rate
		# DELETE later
		# best_ckpt['optimizer']['param_groups'][0]['lr'] = 0.015

	else:
		logfile = "debug.log"
		dname = args.dataset
		if dname == "msvd":
			config_file = "configs/params_msvd.json"
		elif dname == "msrvtt":
			config_file = "configs/params_msrvtt.json"
		elif dname == "vatex":
			config_file = "configs/params_vatex.json" # add later for new dataset
		elif dname == "mini_vatex":
			config_file = "configs/params_vatex.json"
		elif dname == "mid_vatex":
			config_file = "configs/params_vatex.json"
		elif dname == "coco":
			config_file = "configs/params_coco.json"

		config = json.loads(open(config_file, 'r').read())
		config = config['params']
		config['rand_seed'] = args.rand_seed
		config['nclusters'] = args.nclusters
		config['use_one_cap'] = args.use_one_cap
		config['use_one_cap_rand'] = args.use_one_cap_rand
		config['use_pos'] = args.use_pos
		config['lr_byepoch'] = args.lr_byepoch
		config['val_beam_width'] = args.val_beam_width
		config['val_num_samples'] = args.val_num_samples
		config['val_group_size'] = args.val_group_size
		config['diversity_lambda'] = args.dl
		feat_inputs_name = config['visionfeats']+"_z" if args.use_z else config['visionfeats']
		str_psedo = "opsedo" if config['with_psedo']==0 else "wpsedo"
		config['learning_rate'] = args.learning_rate
		config['kl_rate'] = args.kl_rate
		config['cl_loss_prop'] = args.cl_loss_prop
		config['res_dir'] = os.path.join(
			args.res_dir_root, "_".join([feat_inputs_name, dname, str_psedo, 'lr'+str(config['learning_rate']), 'kl'+str(config['kl_rate']), time.strftime("%Y_%m_%d_%H_%M_%S"), args.mark])
		) # add more identification later.
		config['use_z'] = args.use_z
		config['gshuffle'] = bool(1 - args.non_gshuffle)
		config['ckpt_path'] = os.path.join(config['res_dir'], "best_model.ckpt")
		config['dname'] = args.dataset
		config['div_topK'] = args.div_topK
		config['z_indeps'] = args.z_indeps
		config['seq_CVAE'] = args.seq_CVAE
		config['use_CL_loss'] = args.use_CL_loss
		config['SC_latent_dim'] = args.SC_latent_dim
		config['use_SC_loss'] = args.use_SC_loss
		config['visualize'] = args.visualize
		config['scale_o'] = args.scale_o
		

	random.seed(config['rand_seed'])
	np.random.seed(config['rand_seed'])
	torch.manual_seed(config['rand_seed'])

	config = DotMap(config)
	if dname == "mid_vatex":
		config['train_cocoformat'] = "datasets/vatex/mid_vatex_train_cocofmt_0th.json"
		config['val_cocoformat'] = "datasets/vatex/mid_vatex_val_cocofmt_0th.json"
	elif dname == "mini_vatex":
		config['train_cocoformat'] = "datasets/vatex/mini_vatex_train_cocofmt_0th.json"
		config['val_cocoformat'] = "datasets/vatex/mini_vatex_val_cocofmt_0th.json"
	config['max_es_cnt'] = args.max_es_cnt
	if not args.max_epoch is None:
		config['max_epoch'] = args.max_epoch
	if not args.batch_size is None:
		config['batch_size'] = int(args.batch_size)
	if not args.latent_dim_tx is None:
		config['latent_dim_tx'] = int(args.latent_dim_tx)
	if not args.pathToData is None:
		config['pathToData'] = [str(args.pathToData)]
	if not args.best_metrics is None:
		config['best_metrics'] = str(args.best_metrics)
	if args.coco_class:
		config['coco_class'] = args.coco_class
	if not args.pos_div_path is None:
		config['pos_div_path'] = args.pos_div_path
	else:
		config['pos_div_path'] = config['coco_class']
		
	data_path = config['pathToData']
	vocab_path = config['vocab_path']
	coco_class = config['coco_class']
	img_data_path = config['image_data_path']
	batch_size = int(config['batch_size'])
	val_batch_size = int(config['batch_size']) # here batch_size of train and val are the same!
	nseqs_per_video = int(config['nseqs_per_video'])
	maxlength = int(config['maxlength'])
	maxlength_obj = int(config['maxlength_obj'])  
	latent_dim_tx  = int(config['latent_dim_tx'])
	meanimfeats_size = int(config['meanimfeats_size'])
	word_dim = int(config['word_dim'])
	mask_hidden_size = int(config['mask_hidden_size']) # here, confusingly, mask means context.
	max_epochs = int(config['max_epoch'])
	learning_rate = config['learning_rate']
	kl_rate = config['kl_rate']
	with_psedo = bool(config['with_psedo'])
	str_psedo = "wpsedo" if with_psedo else "opsedo"
	best_metrics = config['best_metrics']
	config['bad_ckpt_path'] = os.path.join(config['res_dir'], "bad_model.ckpt")
	print(config['res_dir'])
	config['sc_rate'] = args.sc_rate
	config['sc_gamma_neg'] = args.sc_gamma_neg
	if not os.path.isdir(args.res_dir_root):
		os.mkdir(path=args.res_dir_root)
	if not os.path.isdir(config['res_dir']):
		os.mkdir(path=config['res_dir'])
	
	if os.path.isfile(os.path.join(config['res_dir'], logfile)):
		os.remove(os.path.join(config['res_dir'], logfile))
	config['use_diff_dists_loss'] = args.use_diff_dists_loss
	config['di_rate'] = args.di_rate

	# save the config.
	logging.basicConfig(
		level=getattr(logging, "DEBUG"),
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(os.path.join(config['res_dir'], logfile)), # change later.
			logging.StreamHandler()
			]
	)

	writer = SummaryWriter(config['res_dir'])
	
	# To save important scripts for reproduction.
	import shutil
	if not os.path.isdir(config['res_dir'] +'/scripts'):
		os.mkdir(path=config['res_dir'] + '/scripts')
	shutil.copy('scripts/train.py', config['res_dir']+'/scripts/cp_train.py')
	logger.info("train.py has been copied to %s" % config['res_dir']+'/scripts/cp_train.py')

	shutil.copy('modules/data_loader.py', config['res_dir']+'/scripts/cp_data_loader.py')
	logger.info("data_loader.py has been copied to %s" % config['res_dir']+'/scripts/cp_data_loader.py')

	shutil.copy('modules/textmodules.py', config['res_dir']+'/scripts/cp_textmodules.py')
	logger.info("textmodules.py has been copied to %s" % config['res_dir']+'/scripts/cp_textmodules.py')

	shutil.copy('cococaption/eval_multi.py', config['res_dir']+'/scripts/cp_eval_multi.py')
	logger.info("eval_multi.py has been copied to %s" % config['res_dir']+'/scripts/cp_eval_multi.py')

	logger.info(
		'Input arguments: %s',
		json.dumps(
			config,
			sort_keys=True,
			indent=4
		)
	)


	vocab_class = json.load( open( str(vocab_path),'r'))
	vocab_size = len(vocab_class)
	device = torch.device("cuda:0")

	coco_dataset = COCO_Dataset(data_path,img_data_path,vocab_path,coco_class,nseqs_per_video=nseqs_per_video,mode="train", dataset_name=dname, maxlength=maxlength, maxlength_obj=maxlength_obj, result_dir=config['res_dir'])
	logger.info("the max number of epochs: "+ str(max_epochs))

	val_dataset = COCO_Dataset(data_path,img_data_path,vocab_path,coco_class,nval=args.nval,nseqs_per_video=nseqs_per_video,mode="val",dataset_name=dname, maxlength=maxlength, maxlength_obj=maxlength_obj, result_dir=config['res_dir'])
	logger.info("len of train set: "+str(len(coco_dataset)))
	logger.info("len of val set: "+str(len(val_dataset)))
	nworker = 0
	coco_dataloader = torch.utils.data.DataLoader(coco_dataset, batch_size=batch_size,  shuffle=True, num_workers=nworker, drop_last=True, collate_fn = my_collate) 
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size,  shuffle=False, num_workers=nworker, drop_last=True, collate_fn = my_collate)	

	# Modify later to more general cases.
	if config['use_SC_loss']:
		SC_vocab = json.load(open('datasets/msrvtt/msrvtt_train_SC_3000.json','r'))
		SC_byim = json.load(open('datasets/subsets/verb/msrvtt/msrvtt_kmeans_clusterK5_verb_SC_3k.json'))
		SCix_byim = {k: {kk: [SC_vocab.index(w) for w in vv] for kk, vv in v.items()} for k,v in SC_byim.items()}
		word2SC = {coco_dataset.word2idx[i]: ix for ix, i in enumerate(SC_vocab)}
		# word2SC = {k:int(v in SC_vocab) for k,v in coco_dataset.idx2word.items()}

	if config['use_pos']:
		posFusion = PostFusion(qk_dim=mask_hidden_size, att_dim=256).cuda()
		pos_file = h5py.File(config['use_pos'], 'r')
		pos_z_split = np.array_split(pos_file['context_hidden'][()], len(pos_file['context_hidden'])//config['nclusters'])
		z_id_split = np.array_split(pos_file['vid'][()], len(pos_file['context_hidden'])//config['nclusters'])
		pos_id2z = {int(np.mean(i)): torch.from_numpy(j) for (i, j) in zip(z_id_split, pos_z_split)}

	if config['gshuffle']:
		coco_dataloader_gs = global_shuffle(coco_dataloader, bs=int(batch_size*20))
	else:
		coco_dataloader_gs = coco_dataloader

	iter_per_epoch = len(coco_dataloader_gs)
	logger.info("Num of iteration per epoch: %d" % int(iter_per_epoch))
	max_iters = int(max_epochs * iter_per_epoch)
	logger.info("Num of all iterations: %d" % int(max_iters))
	if config['use_SC_loss']:
		texDec= TextDecoder( 2*latent_dim_tx, vocab_size, encoder_dim=meanimfeats_size, use_z=args.use_z, SC_dim=config['SC_latent_dim'], SC_n_cls=len(SC_vocab)).cuda()
	else:
		texDec= TextDecoder( 2*latent_dim_tx, vocab_size, encoder_dim=meanimfeats_size, use_z=args.use_z).cuda()
	if config['seq_CVAE']:
		texEncMask = TextEncoderMask(word_dim,2*latent_dim_tx, maxlength,meanimfeats_size,vocab_size,mask_hidden_size,use_pos=config['use_pos']).cuda()
		g_prior = GaussPrior(meanimfeats_size=meanimfeats_size, vocab_size=vocab_size, sent_emd_size=word_dim, sent_enc_size=2*latent_dim_tx, max_length=maxlength, z_indeps=args.z_indeps, seq_CVAE=config['seq_CVAE'], scale_o=args.scale_o).cuda()
	else:
		texEncMask = TextEncoderMask(word_dim,latent_dim_tx, maxlength,meanimfeats_size,vocab_size,mask_hidden_size,use_pos=config['use_pos']).cuda() # CHANGE LATER 
		# logger.warning("DEBUDDING, Resume later!!!!")
		# texEncMask = TextEncoderMask(word_dim,latent_dim_tx, maxlength,meanimfeats_size,vocab_size,mask_hidden_size).cuda()
		g_prior = GaussPrior(meanimfeats_size=meanimfeats_size, vocab_size=vocab_size, sent_emd_size=word_dim, sent_enc_size=latent_dim_tx, max_length=maxlength, z_indeps=args.z_indeps, seq_CVAE=config['seq_CVAE'], scale_o=args.scale_o).cuda()
	texEncObj = TextEncoderObj(word_dim,latent_dim_tx, maxlength, meanimfeats_size,vocab_size, mask_hidden_size,z_indeps=config.z_indeps, scale_o=config['scale_o']).cuda()

	

	optimizer = optim.SGD( itertools.chain(texEncMask.parameters(), texEncObj.parameters(), texDec.parameters(), g_prior.parameters() ), 
		lr=learning_rate, momentum=0.9, weight_decay=0.001)#, glow_text_cond.parameters()

	iteration = 0
	init_epoch = True
	epoch = 0

	logger.info("Start Training...")
	best_score = {"set_meteor_hau": 0.0, "oracle_CIDEr": 0.0} # Note the working range!
	best_epoch = 0
	es_cnt = 0

	if args.resume_from:
		# In the process of resuming, remember to add the best accucary!
		logger.info("Resume starting from epoch %d" % (best_ckpt['best_epoch'],))
		epoch = best_ckpt['best_epoch'] + 1
		# iteration = int(epoch * batches_per_epoch) + 1
		best_epoch = best_ckpt['best_epoch']
		if not args.isphase2: # In phrase 2, the first epoch has to be updated.
			best_score = {"set_meteor_hau": best_ckpt['acces']["set_meteor_hau"], "oracle_CIDEr": best_ckpt['acces']["oracle_CIDEr"]}
		best_all_score_acc = best_ckpt['acces']
		best_all_score_div = best_ckpt['divs']
		best_all_losses = best_ckpt['losses']

		g_prior_st = best_ckpt['g_prior_sd']
		g_prior.load_my_state_dict(g_prior_st)
		g_prior = g_prior.cuda()

		txtDecoder_st = best_ckpt['txtDecoder_sd']
		texDec.load_my_state_dict(txtDecoder_st)
		texDec = texDec.cuda()

		txtEncMask_st = best_ckpt['texEncoder_Mask_sd']
		texEncMask.load_my_state_dict(txtEncMask_st)
		texEncMask = texEncMask.cuda()

		txtEncObj_st = best_ckpt['texEncoder_Obj_sd']
		texEncObj.load_my_state_dict(txtEncObj_st)
		texEncObj = texEncObj.cuda()

		optimizer = optim.SGD( itertools.chain(texEncMask.parameters(), texEncObj.parameters(), texDec.parameters(), g_prior.parameters() ), 
		lr=config['learning_rate'], momentum=0.9, weight_decay=0.001)

	print(best_score)
	# sys.exit(0)
	if not args.lr_byepoch:
		logger.info("lr update by batch")
		lr_scheduler = optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=lambda iteration: 1 - iteration / max_iters )
	else:
		logger.info("lr update by epoch")
		lr_scheduler = optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=lambda epoch: 1 - epoch / max_epochs )

	# while iteration + batches_per_epoch < max_iterations:
	while epoch < max_epochs:
	# while True:
		neighbors_list = []
		neighbors_caps = []
		neighbors_caps_masks = []
		oi_pred_sent_emb = []
		if epoch == 1:
			kl_rate = 0.2
		else:
			kl_rate = config['kl_rate'] #debug
			# kl_rate = 1.0
		sc_rate = config['sc_rate']
		# Some metrics to record.
		losses = {"Train":[], "Val":[]}
		losses_nll_tex = {"Train": [], "Val": []}
		losses_kl = {"Train":[], "Val": []}
		losses_neg = {'Train': [], 'Val': []}
		losses_nelbo = {'Train': [], 'Val': []}
		losses_sc = {'Train': [], 'Val': []}
		losses_diff = {'Train': [], 'Val': []}

		mode = "Train" # Here the mode converting is very inconvenient and redunctant!
		# global shuffle
		if mode=='Train' and config['gshuffle'] and epoch != 0 and not config['save_pos']:
			# coco_dataloader = torch.utils.data.DataLoader(coco_dataset, batch_size=batch_size,  shuffle=True, num_workers=nworker, drop_last=True, collate_fn = my_collate)
			logger.debug("Globally shuffling from now!")
			coco_dataloader_gs = global_shuffle(coco_dataloader, bs=int(batch_size*20))
			logger.debug("Globally shuffling done.")
		train_bar = tqdm(coco_dataloader_gs)
		val_bar = tqdm(val_dataloader)

		logger.info("------This is Epoch %d-------", epoch)
		logger.info(os.path.join(config['res_dir'], logfile))
		all_mask_q_means, all_mask_q_logs, all_mask_hiddens, all_obj_q_means, all_obj_q_logs, all_q_z, all_im_idx, all_cluster_idx, all_sents = {},{},{},{},{},{}, {}, {}, {}

		for ind, mybar in enumerate([train_bar, val_bar]):
		# for ind, mybar in enumerate(train_bar):
			if mode == 'Train' and epoch==0:
				mode = 'Val'
				continue
			mode = "Train" if ind == 0 else "Val"
			logger.info("Now, it is the stage of [%s]", mode)
			all_ids = []
			all_cids = []
			all_kl_losses = []
			if mode == "Train":
				n_iters_train = 0 # bug...
				writer.add_scalar("TrainingLearningRate", float(get_lr(optimizer)), epoch)
				
			else:
				n_iters_val = 0
			# 	# torch.cuda.empty_cache() # this function is not suitable!
			# 	# batch_size = 10 if ind == 1 else 32
			if config['save_pos']:			
				all_mask_q_means[mode], all_mask_q_logs[mode] = [],[]
				all_obj_q_means[mode], all_obj_q_logs[mode] = [],[]
				all_mask_hiddens[mode] = []
				all_q_z[mode] = []
				all_im_idx[mode] = [] 
				all_cluster_idx[mode] = []
				all_sents[mode] = []
			for data in mybar:
				if epoch == 0 and mode =="Train":
					continue			
				if mode == "Val":
					n_iters_val += 1
					texEncMask.eval()
					texEncObj.eval()
					texDec.eval()
					for model in [texEncMask, texEncObj, texDec]:
						for para in model.parameters():
							para.requires_grad = False
				else:
					n_iters_train += 1
					texEncMask.train()
					texEncObj.train()
					texDec.train()
					for model in [texEncMask, texEncObj, texDec]:
						for para in model.parameters():
							para.requires_grad = True
			
				optimizer.zero_grad()

				if config['gshuffle'] and mode == 'Train':
					seq_len_per_video = [len(i) for i in data[0]]
					caps = data[0]
					mask_cap = data[1]
					obj_caps = data[2]
					cluster_ids = data[3].numpy()

					bottom_up_features = data[4]
					seq_lengths = data[5]
					im_idx = data[6]
					cap_info = data[7]
				else:
					seq_len_per_video = [len(i) for i in data[0]]
					caps = torch.cat(data[0])
					mask_cap = torch.cat(data[1])
					obj_caps = torch.cat(data[2])
					cluster_ids = torch.cat(data[3]).numpy()

					bottom_up_features = torch.cat([d.unsqueeze(0) for ix,d in enumerate(data[4]) for j in range(seq_len_per_video[ix])])
					seq_lengths = torch.cat(data[5])
					im_idx = np.array([d for ix,d in enumerate(data[6]) for j in range(seq_len_per_video[ix])])
					cap_info = data[7]

				# output for debug
				if args.use_one_cap:
					sid = [0]
					ss = 0
					for l in seq_len_per_video:
						ss += l
						sid.append(ss)
					sid = sid[:-1]
					randid = []
					for s, l in zip(sid, seq_len_per_video):
						if args.use_one_cap_rand:
							randid.append(np.random.randint(s, s+l, 1)[0])
						else:
							randid.append(s)
					seq = caps[randid].to(device)
					mask_seq = mask_cap[randid].to(device)
					obj_seq = obj_caps[randid].to(device)
					bottom_up_features = bottom_up_features[randid].to(device)
					seq_lengths = seq_lengths[randid]

				else:
					seq  = caps.to(device)
					mask_seq  = mask_cap.to(device)
					obj_seq  = obj_caps.to(device)
				

				# bottom_up_features,_,seq,mask_seq,obj_seq, seq_lengths, sort_order = get_properseq(bottom_up_features, None, seq.cpu().numpy(), mask_seq.cpu().numpy(), obj_seq.cpu().numpy(), seq_lengths.cpu().numpy())
				list_v_k = bottom_up_features.float()
				list_v_k = list_v_k[:,:,:meanimfeats_size]

				if mode == "Train":
					if config['use_pos']:
						pos_hiddens = torch.cat([pos_id2z[i].unsqueeze(0) for i in im_idx]).cuda() #(B,k,d)
						# Another modification for adding features from First Stage.
						mask_q_means, mask_q_logs, mask_q_z, mask_hidden = texEncMask(mask_seq[:,1:].cuda(),(seq_lengths-1).cuda(), posHidden=pos_hiddens[range(pos_hiddens.shape[0]), cluster_ids, :])

					else:
						mask_q_means, mask_q_logs, mask_q_z, mask_hidden = texEncMask(mask_seq[:,1:].cuda(),(seq_lengths-1).cuda()) # seems there is not image feature?
								
					if not config['seq_CVAE']:
						obj_seq = obj_seq[:,:20] #maximum top-5 objects 
						obj_seq_len = (torch.ones(obj_seq.size(0),)*20).long().cuda()
						obj_q_means, obj_q_logs, obj_q_z, obj_weights_list = texEncObj(obj_seq.cuda(), list_v_k.cuda(), mask_hidden, mask_q_z, obj_seq_len)
						# to visualize the weights - 
						# video [id] : (obj1, prob1); (obj2, prob2); ...
						if args.visualize: 
							logger.debug("To visualize the weights for different objects in training stage.")
							for i in range(batch_size):
								id = im_idx[i].item()
								obj_prob = []
								for k in range(len(obj_weights_list)):
									for j in range(obj_seq.shape[1]):
										obj_prob.append((coco_dataset.get_i2w()[obj_seq[i,j].item()],round(obj_weights_list[k][i,j].item(),2))) 
									obj_prob = sorted(obj_prob, key=lambda ii:-ii[1])
									logger.debug('[%d] video for step (%d): %s' % (id, k, obj_prob))
							sys.exit(0) # visualize for only one batch.

					if config['seq_CVAE']:
						q_z = mask_q_z
					# elif config['use_pos'] and mode=='Train': # [TEST]: use posteriori from the last stage.
					# 	pos_q_z = torch.cat([pos_id2z[i].unsqueeze(0) for i in im_idx]).cuda() # id must in im_idx
					# 	q_z = posFusion(torch.cat([mask_q_z, obj_q_z, pos_q_z], dim=2))
					else:
						q_z = torch.cat([mask_q_z,obj_q_z],dim=2)

					# To moniter the information of latent
					latent_info = {"mean": round(q_z.mean().item(),3), "std": round(q_z.std().item(),3)}
					writer.add_scalars("Var/Latent", latent_info, iteration)

					if config['save_pos']:
						all_mask_q_means[mode].append(mask_q_means.data.cpu().numpy())
						all_mask_q_logs[mode].append(mask_q_logs.data.cpu().numpy())
						all_mask_hiddens[mode].append(mask_hidden.data.cpu().numpy())
						all_obj_q_means[mode].append(obj_q_means.data.cpu().numpy())
						all_obj_q_logs[mode].append(obj_q_logs.data.cpu().numpy())
						# all_q_z[mode].append(q_z.data.cpu().numpy())
						all_im_idx[mode].append(im_idx)
						all_cluster_idx[mode].append(cluster_ids)
						all_sents[mode] += coco_dataset.arr2lang(caps.cpu().numpy())
						continue
				
				else: # for eval
					# if epoch == 0:
					mask_q_means, mask_q_logs, mask_q_z, mask_hidden = None, None, None, None
					obj_q_means, obj_q_logs, obj_q_z = None, None, None
					# For now, wrong placeholder for val
					q_z = torch.ones(caps.shape[0], maxlength, int(2*latent_dim_tx))
				if mode == "Train":
					if config['seq_CVAE']:
						kl_loss_all  =  g_prior(obj_enc=list_v_k.cuda(), x=seq.cuda(), p_z_t_1=None, hiddens=None, 
											q_means_mask=mask_q_means, q_logs_mask=mask_q_logs, q_z_mask=mask_q_z, 
											q_means_obj=None, q_logs_obj=None, q_z_obj=None, train=(mode == 'Train'), reverse=False)
					else:
						kl_loss_all, p_z_info =  g_prior(obj_enc=list_v_k.cuda(), x=seq.cuda(), p_z_t_1=None, hiddens=None, 
											q_means_mask=mask_q_means, q_logs_mask=mask_q_logs, q_z_mask=mask_q_z, 
											q_means_obj=obj_q_means, q_logs_obj=obj_q_logs, q_z_obj=obj_q_z, train=(mode == 'Train'), reverse=False) # Here it is not that straightforward, should be f(prior|posterior).

					# writer.add_scalars("Var/Prior", p_z_info, iteration)
				# if mode == "Val" and epoch != 0:
				# 	q_z = p_z # for calculating the nll loss for test set, for the sake of simplicy, I didn't change the name. [WARNIGN] BAD CODE!

				seq_lengths = seq_lengths - 1
				if config['use_SC_loss']:
					logp, alphas, attn_ent, SC_logits= texDec(seq.long().cuda(),None,q_z.cuda(),list_v_k.cuda(),seq_lengths.cuda(),args.val_beam_width)
				else:
					logp, alphas, attn_ent = texDec(seq.long().cuda(),None,q_z.cuda(),list_v_k.cuda(),seq_lengths.cuda(),args.val_beam_width)

				tokens_mask = seq != 0 # Loss for valid sequences ignoring pads
				# Debug, delete later
				if logp.sum().isnan().item():
					print("Stop here.")
				
				# try:
				if config['use_CL_loss']:
					# logger.warning("You are using CL loss with more memory assumption!")
					pos_loss, avg_neg_loss = get_CL_loss(logp, seq[:, 1:].contiguous(), tokens_mask[:, 1:].contiguous(), seq_len_per_video, mode=mode, cluster_ids=cluster_ids, im_idx=im_idx)					
					avg_neg_loss = - config['cl_loss_prop'] * avg_neg_loss
					nll_tex = pos_loss + avg_neg_loss
					losses_neg[mode].append(avg_neg_loss.item())
				else:
					nll_tex = torch.mean(get_loss(logp, seq[:, 1:].contiguous(), tokens_mask[:, 1:].contiguous()))

				if mode == "Val":
					kl_loss = torch.zeros(1)
				else:
					kl_loss = torch.mean(kl_loss_all)
					all_ids.append(im_idx)
					all_cids.append(cluster_ids)
					all_kl_losses.append(kl_loss_all)
					# if len(all_ids) >= 5:
					# 	break

				
				if args.use_z:

					if mode == "Train" and not config['seq_CVAE']:
						diff_kls_loss_all = DiffDistsLoss(mask_q_means, mask_q_logs, obj_q_means, obj_q_logs, cluster_ids, im_idx, seq_lengths)		
						diff_kls_loss = torch.mean(diff_kls_loss_all)
					else:
						kl_loss_all = torch.zeros(config['batch_size']).cuda()
						diff_kls_loss_all = torch.zeros(config['batch_size']).cuda()
						diff_kls_loss = torch.tensor(.0)

					# loss = 1.0*nll_tex.cuda() + 1.0*kl_rate*kl_loss.cuda() - config['di_rate']*((diff_kls_loss_all+1e-4).log()).mean()
					if ((diff_kls_loss_all+1e-4).log()).mean().isnan().item():
						print("Here.")
					# NOTE: why not output this?
					losses_diff[mode].append((diff_kls_loss_all+1.001).log().mean().item())

					if config['use_SC_loss']:
						# NOTE: modify later
						
						sc_loss = torch.mean(get_SC_loss(SC_logits, seq[:, 1:].contiguous(), im_idx, cluster_ids, config['sc_gamma_neg']))
						loss = 1.0*nll_tex.cuda() + 1.0*kl_rate*kl_loss.cuda() + sc_rate*sc_loss.cuda()
						losses_sc[mode].append(sc_loss.item()) # Should be unweighted!

					elif config['use_diff_dists_loss']:
						loss = 1.0*nll_tex.cuda() + 1.0*kl_rate*(kl_loss_all.cuda() / config['di_rate']*(diff_kls_loss_all+1.001).log()).mean()
					else:
						loss = 1.0*nll_tex.cuda() + 1.0*kl_rate*kl_loss.cuda()
					# try:
					if loss.isnan().item():
						print("Stop here & ckpt has been saved for debugging.")
						torch.save(checkpoint, config['bad_ckpt_path'])
						exit(0)
					# except:
					# 	pass
				else:
					loss = 1.0*nll_tex.cuda()


				mybar.set_description(mode+' loss %.2f | Epoch %d -- Iteration ' % (loss.item(),epoch))

				losses[mode].append(loss.item())
				losses_nll_tex[mode].append(nll_tex.item()) # These two items are to show the variations of respective parts.
				losses_kl[mode].append(kl_loss.item()) # Should be unweighted!
				losses_nelbo[mode].append(nll_tex.item() + kl_loss.item())

				if mode == "Train":
					loss.backward()
					nn.utils.clip_grad_norm_(texEncMask.parameters(), 12.5)
					nn.utils.clip_grad_norm_(texEncObj.parameters(), 12.5)
					nn.utils.clip_grad_norm_(g_prior.parameters(), 12.5)
					nn.utils.clip_grad_norm_(texDec.parameters(), 12.5)
					
					optimizer.step()
					iteration += 1
					if not args.lr_byepoch:
						lr_scheduler.step(iteration)
					else:
						lr_scheduler.step(epoch)
				if args.debug:
					break

		init_epoch = True # no nn!
		
		if config['save_pos']:
			logger.info("To save the posteriori probability, only one epoch is enough!")
			logger.info(">>>>>>>Save Posteriori>>>>>>")

			for m in ['Train']:#, 'Val']:
				pos_data = h5py.File(config['res_dir']+'/sample_z_pos_pred_%s.h5'%m, 'w')
				vid_data = np.concatenate(all_im_idx[m])
				sorted_ix = np.argsort(vid_data)
				pos_data.create_dataset('vid', data=np.concatenate(all_im_idx[m])[sorted_ix])
				pos_data.create_dataset('cid', data=np.concatenate(all_cluster_idx[m])[sorted_ix])
				pos_data.create_dataset('context_mean', data=np.concatenate(all_mask_q_means[m])[sorted_ix] )
				pos_data.create_dataset('context_logstd', data=np.concatenate(all_mask_q_logs[m])[sorted_ix] )
				pos_data.create_dataset('context_hidden', data=np.concatenate(all_mask_hiddens[m])[sorted_ix] )
				pos_data.create_dataset('verb_mean', data=np.concatenate(all_obj_q_means[m])[sorted_ix] )
				pos_data.create_dataset('verb_logstd', data=np.concatenate(all_obj_q_logs[m])[sorted_ix] )
				# pos_data.create_dataset('q_z', data=np.concatenate(all_q_z[m])[sorted_ix] )
				gt_sents = [all_sents[m][i].encode("ascii", "ignore") for i in sorted_ix]
				pos_data.create_dataset('sents', (len(gt_sents),1),'S100', gt_sents)
				logger.info(config['res_dir']+'/sample_z_pos_pred_%s.h5'%m)

			logger.info("Save done.")
			sys.exit(0)

		if epoch == 0:
			losses['Train'] = [.0]
			losses_nll_tex['Train'] = [.0]
			losses_kl['Train'] = [.0]
			losses_sc['Train'] = [.0]
			losses_diff['Train'] = [.0]
			losses_neg['Train'] = [.0]
			losses_nelbo['Train'] = [.0]
			n_iters_train = 1
		train_avg_loss = sum(losses['Train']) / n_iters_train
		train_avg_loss_tex = sum(losses_nll_tex['Train']) / n_iters_train
		train_avg_loss_kl = sum(losses_kl['Train']) / n_iters_train
		train_avg_loss_sc = sum(losses_sc['Train']) / n_iters_train
		train_avg_loss_diff = sum(losses_diff['Train']) / n_iters_train
		train_avg_loss_neg = sum(losses_neg['Train']) / n_iters_train
		train_avg_loss_nelbo = sum(losses_nelbo['Train']) / n_iters_train

		val_avg_loss = sum(losses['Val']) / n_iters_val
		val_avg_loss_tex = sum(losses_nll_tex['Val']) / n_iters_val
		val_avg_loss_kl = sum(losses_kl['Val']) / n_iters_val
		val_avg_loss_sc = sum(losses_sc['Val']) / n_iters_val
		val_avg_loss_diff = sum(losses_diff['Val']) / n_iters_val
		val_avg_loss_neg = sum(losses_neg['Val']) / n_iters_val
		val_avg_loss_nelbo = sum(losses_nelbo['Val']) / n_iters_val

		writer.add_scalars("Loss", {'train_all_loss':train_avg_loss}, epoch)
		writer.add_scalars("Loss", {'train_text_loss': train_avg_loss_tex}, epoch)
		writer.add_scalars("Loss", {'train_kl_loss': train_avg_loss_kl}, epoch)
		writer.add_scalars("Loss", {'train_neg_loss': train_avg_loss_neg}, epoch)
		writer.add_scalars("Loss", {'train_neg_loss_nelbo': train_avg_loss_nelbo}, epoch)
		writer.add_scalars("Loss", {'train_sc_loss': train_avg_loss_sc}, epoch)
		writer.add_scalars("Loss", {'train_diff_loss': train_avg_loss_diff}, epoch)

		writer.add_scalars("val_Loss", {'val_text_loss': val_avg_loss_tex}, epoch)

		checkpoint = {
		'texEncoder_Mask_sd': texEncMask.state_dict(),
		'texEncoder_Obj_sd': texEncObj.state_dict(),
		'txtDecoder_sd': texDec.state_dict(),
		'g_prior_sd': g_prior.state_dict(),
		'optimizer': optimizer.state_dict(),
		'best_epoch': epoch,
		'model_config': config,
		'losses':
		{
		'best_train_nelbo_loss': train_avg_loss_nelbo,
		'best_train_all_loss': train_avg_loss,
		'best_train_nll_loss': train_avg_loss_tex,
		'best_train_kl_loss': train_avg_loss_kl,
		'best_train_neg_loss': train_avg_loss_neg,
		'best_train_diff_loss': train_avg_loss_diff,
		'best_val_nelbo_loss': val_avg_loss_nelbo,
		'best_val_all_loss': val_avg_loss,
		'best_val_nll_loss': val_avg_loss_tex,
		'best_val_kl_loss': val_avg_loss_kl,
		'best_val_neg_loss': val_avg_loss_neg,}
		}

		if epoch == 0:
			epoch += 1
			continue

		logger.info(
					'Values for various Losses: %s',
					json.dumps(
						checkpoint['losses'],
						sort_keys=True,
						indent=4
					)
				)
		logger.info(config['res_dir'])
		out, _, out_div = eval(val_dataloader, checkpoint, config, logger, write_result=False, debug=args.debug, pred_rr_path=os.path.join(config['res_dir'], "pred_rerank_con.json"))
		out, out_div = out[0], out_div[0]
		logger.info(config['res_dir'])
		writer.add_scalars("Val/Acc", out['overall'], epoch)
		checkpoint['acces'] = out['overall']
		checkpoint['divs'] = out_div['overall']

		div_dict_percent = {k:v for k,v in out_div['overall'].items() if k in ['Div1', 'Div2', 
									'lUni', 'gUni', 'lUniv', 'gUniv',  'mBLeu_4']}
		div_dict_number = {k:v for k,v in out_div['overall'].items() if k in ['gDiv1', 'Novsents']}
		writer.add_scalars("Val/Div_percent", div_dict_percent, epoch)
		# writer.add_scalars("Val/Div_number", div_dict_number, epoch)

		# two end lines: not improved or last line
		if out['overall'][best_metrics] > best_score[best_metrics]:
			best_score[best_metrics] = out['overall'][best_metrics]
			best_epoch = epoch
			best_all_score_acc = out['overall']
			best_all_score_div = out_div['overall']
			best_all_losses = checkpoint['losses']
			es_cnt = 0
			config['ckpt_path'] = os.path.join(config['res_dir'], "best_model.ckpt")
			save_pred_path = open(os.path.join(config['res_dir'], "best_val_pred.json"),"w")
			# json.dump(pred_file, save_pred_path)
			logger.info("best prediction file of validation has been saved to %s" % save_pred_path)
			if not args.debug:
				torch.save(checkpoint, config['ckpt_path'])
			logger.info("The checkpoint file has been updated.")
		else:
			es_cnt += 1
			if es_cnt > args.max_es_cnt:  # early stop
				logger.info("Early stop at {} with {} {}".format(epoch, best_metrics, out['overall'][best_metrics]))
				logger.info("The best {} is {} of epoch {}".format(best_metrics, best_score[best_metrics], best_epoch))
				logger.info("--------------------------")
				logger.info(
					'Output Metrics of Acc: %s',
					json.dumps(
						best_all_score_acc,
						sort_keys=True,
						indent=4
					)
				)
				logger.info(
					'Output Metrics of Div: %s',
					json.dumps(
						best_all_score_div,
						sort_keys=True,
						indent=4
					)
				)
				logger.info(
					'Output Metrics of *Unweighted* Loss: %s',
					json.dumps(
						best_all_losses,
						sort_keys=True,
						indent=4
					)
				)
			
				logger.info("---Now, we evaluate the full *test* data---")

				test_dataset = COCO_Dataset(data_path,img_data_path,vocab_path,coco_class,nval=3000,nseqs_per_video=nseqs_per_video,mode="test",dataset_name=dname, maxlength=maxlength, maxlength_obj=maxlength_obj, result_dir=config['res_dir'])

				test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size,  shuffle=False, num_workers=nworker, drop_last=True, collate_fn = my_collate)

				config['val_num_samples'] = 20
				config['ckpt_path'] = os.path.join(config['res_dir'], "best_model.ckpt")
				checkpoint = torch.load(open(config['ckpt_path'], 'rb')) # [debug] from the best ckpt instead of the last one!
				out, _, out_div = eval(test_dataloader, checkpoint, config, logger, write_result=False, mode='test', debug=args.debug, pred_rr_path=os.path.join(config['res_dir'], "test_pred_rerank_con.json"))
				out, out_div = out[0], out_div[0]
				append2csv(config, logger, best_all_score_acc, best_all_score_div, best_epoch=best_epoch)
				logger.info("The log file is: %s" % os.path.join(config['res_dir'], logfile))
				logger.info("The pred json file is: %s" % os.path.join(config['res_dir'], "best_val_pred.json"))
				logger.info("The best ckpt file is: %s" % os.path.join(config['res_dir'], "best_model.ckpt"))

				break


		if with_psedo:
			init_epoch = False
		epoch += 1 # max iteration 70000? And without early stopping?

		if epoch == max_epochs:
			logger.info("---Now, we evaluate the full *test* data---")

			test_dataset = COCO_Dataset(data_path,img_data_path,vocab_path,coco_class,nval=3000,nseqs_per_video=nseqs_per_video,mode="test",dataset_name=dname, maxlength=maxlength, maxlength_obj=maxlength_obj, result_dir=config['res_dir'])
			config['val_num_samples'] = 20
			test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size,  shuffle=False, num_workers=nworker, drop_last=True, collate_fn = my_collate)
			checkpoint = torch.load(open(config['ckpt_path'], 'rb')) # [debug] from the best ckpt instead of the last one!
			out, _, out_div = eval(test_dataloader, checkpoint, config, logger, write_result=False, mode='test', debug=args.debug, pred_rr_path=os.path.join(config['res_dir'], "test_pred_rerank_con.json"))
			out, out_div = out[0], out_div[0]
			# append2csv(config, logger, best_all_score_acc, best_all_score_div, best_epoch=best_epoch)
			logger.info("The best epoch is: %d" % best_epoch)
			logger.info("The log file is: %s" % os.path.join(config['res_dir'], logfile))
			logger.info("The pred json file is: %s" % os.path.join(config['res_dir'], "best_val_pred.json"))
			logger.info("The best ckpt file is: %s" % os.path.join(config['res_dir'], "best_model.ckpt"))

			logger.info(">>>>>>>>>>>>>>>>>>>>>>>>")
			logger.info("All epochs have been used up~")
		if args.save_all_ckpts:
			if not os.path.isdir(config['res_dir']+"/ckpts"):
				os.mkdir(path=config['res_dir']+"/ckpts")
			torch.save(checkpoint, os.path.join(config['res_dir'], "ckpts/model_%d.ckpt"%epoch) )