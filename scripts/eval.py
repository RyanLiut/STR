from __future__ import print_function
import os
import random
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import itertools
from tqdm import tqdm
import pickle
import json
import pandas as pd
import sys
import utils
sys.path.insert(0, './cococaption')
sys.path.insert(0,'./modules')
from build_vocab_coco import Vocabulary
sys.path.insert(0,'./')
from modules.data_loader import COCO_Dataset, my_collate
from modules.textmodules import TextDecoder, GaussPrior
from modules.utils import masked_mean
from cococaption.eval_multi import eval_oracle, eval_spice_n, eval_div_stats, eval_single, eval_setAcc
from dotmap import DotMap
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
logger = logging.getLogger(__name__)
import h5py
import time
import shutil
# from parrot import Parrot

def loglikehihoods_to_str(vocab_wordlist, loglikelihood_seq, length ):
	curr_pred = [vocab_wordlist[int(p)] for p in loglikelihood_seq][0:length]
	curr_pred_str = []
	for curr_word in curr_pred:
		if (curr_word == "."):
			curr_pred_str += [curr_word]
			break;
		elif curr_word == "<end>":
			break;
		elif (curr_word != "<start>") and (curr_word != "<end>") and (curr_word != "<pad>") and (curr_word != "<unk>"):
			curr_pred_str += [curr_word] #+ ' ';	

	return curr_pred_str


def get_round(mylist):
	return [round(i,3) for i in mylist]



def append2csv(config, logger, best_all_score_acc, best_single_score_acc, best_all_score_div, mark='-', best_epoch=None, pred_path=None):
	# Automatically save the results to csv
	df = pd.DataFrame() # change it later to append
	df['dataset'] = [config['dname']]
	df['mark'] = [mark]
	# df['best_epoch'] = [best_epoch]
	df['#sample(B_S_G)'] = [str(config['val_beam_width'])+'_'+str(config['val_num_samples'])+'_'+str(config['val_group_size'])]

	for k,v in best_all_score_acc.items():
		if not k in ["all_time", "oracle_Bleu_1", "oracle_Bleu_1_mean", "oracle_Bleu_2", "oracle_Bleu_2_mean", "oracle_Bleu_3", "oracle_Bleu_3_mean"]:
			df[k] = [round(v*100, 2)]

	for k,v in best_single_score_acc.items():
		df[k] = [round(v*100, 2)]

	for k,v in best_all_score_div.items():
		if k[:2] != "mB": # only record mix_mBleu score.
			df[k] = [round(v*100, 2)]

	# df['all_time'] = round(best_all_score_acc['all_time'],2)
	df['folder_name'] = [config['res_dir'][len('myresults/'):]]

	if os.path.exists(global_result_path):
		df_g = pd.read_csv(global_result_path)
		if len(list(df_g.columns)) == len(list(df.columns)):
			df_g = df_g.append(df, ignore_index=True)
		else:
			df_g = df
		logger.debug("The result path has been updated to %s"%global_result_path)
	else:
		df_g = df
		logger.debug("The result path has been updated %s"%global_result_path)
	df_g.to_csv(global_result_path, index=False)

def rerank(pred, topK=5):
	# rerank the prediction file
	logger.info("Now, it is reranking by len.")
	id2cap = {}
	for i in pred:
		if i['image_id'] not in id2cap:
			id2cap[i['image_id']] = []
		id2cap[i['image_id']].append(i['caption'])
	inx = []
	def rankbylen(mylist, topK):
		num_list = [-len(i.split()) for i in mylist]
		inx = np.argsort(num_list)[::len(num_list)//topK+1][:topK]
		return [mylist[i] for i in inx]
	id2cap_rr = {k:rankbylen(v,topK) for k,v in id2cap.items()}
	id2cap_rr =[{'image_id': k, 'caption': i} for k,v in id2cap_rr.items() for i in v]
	logger.info("The size of caption set after reranking by len is %s" % str(len(inx)))
	
	return id2cap_rr

def rerank_bymetrics(annFile, pred, metrics=['CIDEr'], root=None):
	id2cap = {}
	for d in pred:
		id2cap[d['image_id']] = id2cap.get(d['image_id'], []) + [d]
	for m in metrics:
		id2cap_bym = {}
		for k,v in id2cap.items():
			id2cap_bym[k] = sorted(v, key=lambda i: -i['scores'][m])
		pred_bym = [{'image_id': k, 'caption': v[0]['caption']} for k,v in id2cap_bym.items()]
		json.dump(pred_bym, open(root+'/temp.json','w'))
		out_bym = eval_single(annFile, root+'/temp.json')
		logger.info('\n------------%s-------------', m)
		logger.info(
			'Output metrics from: %s',
				json.dumps(
				out_bym,
				sort_keys=True,
				indent=4
			)
		)

def consensus_rerank(pred_results, Metric='CIDEr', topK=5):
	'''
	Consensus reranking.
	Reranking the predicted captions in a set using the GT captions from the nearest image in the training set
	Notice: some video ids in VATEX cannot be available and need to be removed.
	'''
	logger.info("Now, it is consensus reranking using CLIP2Video.")
	id2cap = {}
	for i in pred_results:
		if i['image_id'] not in id2cap:
			id2cap[i['image_id']] = []
		id2cap[i['image_id']].append((i['caption'], i['scores'][Metric]))

	id2cap_rr = {k:sorted(v, key=lambda i: i[1], reverse=True)[:topK] for k,v in id2cap.items()}
	# output ciderscore to debug. Delete it later.
	id2cap_rr =[{'image_id': k, 'caption': i[0], Metric: i[1]} for k,v in id2cap_rr.items() for i in v]
	
	return id2cap_rr

def retrieval_format(pred_n_file, selected_file, name2id_file):
	'''
	Convert the list-format pred_n into the dictionary-format {id, {{'videoID':x, 'enCap':[c1,c2...]}}}
	To adapt to the retrieval ann_file
	Notice: only VATEX is such format for now
	'''
	pred_n = json.load(open(pred_n_file, 'r'))
	selected_ids = pd.read_csv(selected_file, header=None)[0].to_list()
	id2name = {row['id']:row['name'] for _,row in pd.read_csv(name2id_file).iterrows()}

	capsById = {}
	for d in pred_n:
		if id2name[d['image_id']] in selected_ids:
			capsById[id2name[d['image_id']]] = capsById.get(id2name[d['image_id']], []) + [d['caption']]
	
	pred_ann = {}
	for i in capsById:
		pred_ann[i] = {}
		pred_ann[i]['videoID'] = i
		pred_ann[i]['enCap'] = capsById[i]

	print("The length of pred_ann is %d" % len(pred_ann))
	json.dump(pred_ann, open(pred_n_file[:-5]+'_for_ret.json', 'w'))
	print(pred_n_file[:-5]+'_for_ret.json')
	
	return pred_ann

def paraphraseAug(pred_result, numV=4):
	'''
	To augment sentences via their paraphrase using *parrot*
	'''
	parrot = Parrot(model_tag="/disk2/liuzhu/resources/hugging/parrot_paraphraser_on_T5/")
	pred_aug = []

	for i in tqdm(pred_result):
		# print(i['image_id'])
		item = [{"image_id": i['image_id'], "caption": j[0]} for j in parrot.augment(input_phrase=i['caption'], use_gpu=True,do_diverse=True, max_return_phrases=20)[:numV]]
		pred_aug += item

	return pred_aug

def eval(val_dataloader, ckpt, config, logger, write_result=False, mode="val", debug=False, pred_path=None, pred_rr_path=None, mul_runs=1, post_z=None, use_pos=False, paraAug=False, external_file=None):
	out_all_runs, out_div_rerank_all_runs, out_acc_rerank_all_runs = [], [], []
	snum = int(config.val_num_samples * config.val_beam_width * config.val_group_size)
	
	for n_run in range(mul_runs):
		logger.info(">>>This is run %d" % n_run)
		logger.info("Load vocab file from %s.", config.vocab_path)
		vocab_wordlist = json.load(open(config.vocab_path, "r"))
		assert config.vocab_size == len(vocab_wordlist)
		texDec= TextDecoder(2*config.latent_dim_tx, 
							config.vocab_size, encoder_dim=config.meanimfeats_size,use_z=config.use_z, debug=debug, logger=logger).cuda()
		if config['seq_CVAE']:
			g_prior = GaussPrior(config.meanimfeats_size, config.vocab_size, config.word_dim, 
								2*config.latent_dim_tx, config.maxlength, z_indeps=config['z_indeps'], seq_CVAE=config['seq_CVAE'], scale_o=config['scale_o']).cuda()
		else:
			g_prior = GaussPrior(config.meanimfeats_size, config.vocab_size, config.word_dim, 
							config.latent_dim_tx, config.maxlength, z_indeps=config['z_indeps'], seq_CVAE=config['seq_CVAE'], scale_o=config['scale_o']).cuda()

		g_prior_st = ckpt['g_prior_sd']
		g_prior.load_my_state_dict(g_prior_st)
		g_prior = g_prior.cuda()

		txtDecoder_st = ckpt['txtDecoder_sd']
		texDec.load_my_state_dict(txtDecoder_st)
		texDec = texDec.cuda()
					
		pred_results = []
		# pred_results = json.load(open('myresultsf/SAAT_msrvtt/beams_result_msrvtt.json','r'))['annotations']
		pred_z = [] # save all the sampling z

		vatex_no_avail_ids = json.load(open('datasets/vatex/no_available_%s_ids.json'%mode, 'r'))
		batches_all_sample_zs = []
		batches_all_h_priorm_list = []
		batches_all_h_prioro_list = []
		batches_all_id_list = []
		batches_all_caption_list = []

		if use_pos: 
			logger.info(">>>loading posteriori beginning.>>>")
			pos_info = h5py.File(use_pos, 'r')
			pos_info_id2z = {}
			for i,id in enumerate(pos_info['vid'][:]):
				if id not in pos_info_id2z:
					pos_info_id2z[id] = []
				pos_info_id2z[id].append(( pos_info['q_z'][i], pos_info['sents'][i][0].decode('UTF-8') ))
			if config['dname'] == "msrvtt":
				logger.warning("For msrvtt, the *166th* video is missing!")
				pos_info_id2z[166] = pos_info_id2z[167]

			logger.info(">>>loading posteriori done.>>>")
   
		if not paraAug: # only for the external validation
			for i,data in tqdm(enumerate(val_dataloader)):
				with torch.no_grad():
					bottom_up_features = torch.cat([i.unsqueeze(0) for i in data[4]])
					im_idx = np.array(data[-2])
					
					seq_in = None

					mask_dec_in = bottom_up_features.float()
					mask_dec_in = mask_dec_in[:,:,:config.meanimfeats_size]
					#print('bottom_up_classes ',bottom_up_classes[0].view(-1))
					obj_enc_mask = torch.sum(torch.abs(mask_dec_in), dim=-1) > 0
					#print('obj_enc_mask ',obj_enc_mask[0])
					meanimfeats = masked_mean( mask_dec_in, obj_enc_mask.unsqueeze(-1) ,dim=1)
					all_preds_inf_latent = []
					all_sample_zs = {}
					all_h_priorm_list = {}
					all_h_prioro_list = {}

					start_time = time.time()
					for t_idx in range(config.val_num_samples):
						# The speed may be influeced here. O(test_samples * beam_width)
						# logger.debug('-------------%s------------' % str(t_idx))
						# h_priorm_list : max_len * (1, BS, d, bw)
						if use_pos:
							if t_idx == len(pos_info_id2z[i]):
								break
							try:
								post_z = np.stack([pos_info_id2z[i][t_idx][0] for i in im_idx])
							except:
								print(i, t_idx)
						preds, sample_zs, h_priorm_list, h_prioro_list = texDec(seq_in,None, None,mask_dec_in.cuda(),config.maxlength,config.val_beam_width, config.val_group_size, config.diversity_lambda, prior_model=g_prior, train = False, post_z=post_z)

						for b in range(config.val_beam_width*config.val_group_size):
							all_preds_inf_latent.append(preds[:,:,b])
							# for t in range(len(sample_zs)):
							# 	if t not in all_sample_zs:
							# 		all_sample_zs[t] = []
							# 	all_sample_zs[t].append(sample_zs[t][...,b])

							# for t in range(len(h_priorm_list)):
							# 	if t not in all_h_priorm_list:
							# 		all_h_priorm_list[t] = []
							# 	all_h_priorm_list[t].append(h_priorm_list[t][...,b])

							# for t in range(len(h_prioro_list)):
							# 	if t not in all_h_prioro_list:
							# 		all_h_prioro_list[t] = []
							# 	all_h_prioro_list[t].append(h_prioro_list[t][...,b])
					delta_time = time.time()-start_time

					all_preds_inf_latent = np.array(all_preds_inf_latent)
					all_preds_inf_latent = np.transpose(all_preds_inf_latent,(1,0,2))

					# # maxStep * B * sample_id * d 
					# all_sample_zs = torch.cat([torch.cat(v).permute(1,0,2).unsqueeze(2) for _,v in all_sample_zs.items()], dim=2)
					# all_h_priorm_list = torch.cat([torch.cat(v).permute(1,0,2).unsqueeze(2) for _,v in all_h_priorm_list.items()], dim=2)
					# all_h_prioro_list = torch.cat([torch.cat(v).permute(1,0,2).unsqueeze(2) for _,v in all_h_prioro_list.items()], dim=2)

					# batches_all_sample_zs.append(all_sample_zs)
					# batches_all_h_priorm_list.append(all_h_priorm_list)
					# batches_all_h_prioro_list.append(all_h_prioro_list)

					im_id_list = []
					cap_list = []
					for j in range(config.batch_size):
						if config.dname == "vatex" and int(im_idx[j]) in vatex_no_avail_ids:
							logger.info("Video %d is not available." % int(im_idx[j]))
							continue
						z_items = []
						h_priorm_items = []
						h_prioro_items = []
						for test_sample_idx in range(config.val_beam_width*config.val_num_samples*config.val_group_size):
							# logger.debug('-------------%s------------' % str(test_sample_idx))
							curr_pred_str = loglikehihoods_to_str(vocab_wordlist, 
							all_preds_inf_latent[j,test_sample_idx], config.maxlength-1)
							strlen = len(curr_pred_str)
							# print(strlen) # debug
							curr_pred_str = ' '.join(curr_pred_str)
							entry = {"image_id":int(im_idx[j]), "caption": curr_pred_str}
							im_id_list.append(int(im_idx[j]))
							cap_list.append(curr_pred_str)
							pred_results += [entry]
							if use_pos:
								logger.debug('*posteriori* %s: %s' %
								(entry['image_id'], pos_info_id2z[im_idx[j]][test_sample_idx][1] ))
							logger.debug('[%d] video %s: %s' %
								(j, entry['image_id'], entry['caption']))
							# UPDATE: if BS, then only the first caption (best NLL)
							# if config.val_num_samples == 1:
							# 	break
						# 	z_items.append(all_sample_zs.mean(dim=0)[j,test_sample_idx].tolist())
						# 	h_priorm_items.append(all_h_priorm_list.mean(dim=0)[j,test_sample_idx].tolist())
						# 	h_prioro_items.append(all_h_prioro_list.mean(dim=0)[j,test_sample_idx].tolist())
						# 	for k,v in all_sample_zs.items():
						# 		logger.debug('%s step; z sample: %s' % (k, get_round(v[j,test_sample_idx].tolist())) )
						# pred_z.append({'video_id': int(im_idx[j]), 'z': z_items, 'h_priorm': h_priorm_items, 'h_prioro': h_prioro_items})

				if debug:
					logger.info("In the stage of debug, thus only one batch is output!")
					logger.info("Stop.") # ---CHANGE
					break
					# sys.exit(0)
				# batches_all_id_list += im_id_list
				# batches_all_caption_list += cap_list

			# batches_all_sample_zs = torch.cat(batches_all_sample_zs)
			# batches_all_h_priorm_list = torch.cat(batches_all_h_priorm_list)
			# batches_all_h_prioro_list = torch.cat(batches_all_h_prioro_list)

			# # json.dump(pred_z, open(config['res_dir']+'/sample_z_4.json', 'w'))
			# context_temp = (batches_all_h_priorm_list @ g_prior_st['fc_m_out.weight'].cuda().T + g_prior_st['fc_m_out.bias'].cuda())[:,0,...]
			# context_mean = context_temp[...,:config.latent_dim_tx]
			# context_logstd = context_temp[...,config.latent_dim_tx:]

			# verb_temp = (batches_all_h_prioro_list @ g_prior_st['fc_o_out.weight'].cuda().T + g_prior_st['fc_o_out.bias'].cuda())[:,0,...]
			# verb_mean = verb_temp[...,:config.latent_dim_tx]
			# verb_logstd = verb_temp[...,config.latent_dim_tx:]

			# h5 = h5py.File(config['res_dir']+'/sample_z_train.h5', 'w')
			# h5.create_dataset('all_sample_zs', data=batches_all_sample_zs.cpu().numpy())
			# h5.create_dataset('context_mean', data=context_mean.cpu().numpy())
			# h5.create_dataset('context_logstd', data=context_logstd.cpu().numpy())
			# h5.create_dataset('verb_mean', data=verb_mean.cpu().numpy())
			# h5.create_dataset('verb_logstd', data=verb_logstd.cpu().numpy())
			# # h5.create_dataset('all_h_priorm_list', data=all_h_priorm_list.cpu().numpy())
			# # h5.create_dataset('all_h_prioro_list', data=all_h_prioro_list.cpu().numpy())
			# h5.create_dataset('vid', data=np.array(im_id_list))
			# cap_list = [n.encode("ascii", "ignore") for n in cap_list]
			# h5.create_dataset('captions', data=cap_list)
			# h5.close()
			# sys.exit(0)
			if pred_path:
				json.dump(pred_results, open(pred_path, "w"))
			# sys.exit(0) # DEBUG!!

		annFile = config[mode+'_cocoformat']

		if paraAug:
			# logger.info(">>>Augment sentences by parrot paraphrases")
			# pred_results = paraphraseAug(pred_results, numV = 4)
			# if pred_path:
			# 	json.dump(pred_results, open(pred_path[:-5]+'_para_r%d.json'%n_run, "w"))
			logger.info(">>>Only use the external file>>>")
			pred_results = json.load(open(args.external_file))

		# sys.exit(0)
		
		# logger.info(">>>set evaluate by GT in run %d>>>" % n_run)
		
		# (out_a, out_b, out_c, out_d), set_all = eval_setAcc(json.load(open(annFile,'r'))['annotations'], pred_results, df_file_path=config['df_file_path'], metric="METEOR")

		logger.info(">>>oracle evaluate by GT in run %d>>>" % n_run)
		
		# print(annFile)
		# json.dump(pred_results, open("debug_pred_results.json","w"))
		out = eval_oracle( annFile, pred_results, config['res_dir'], 'ours', 'val')
		
		# logger.info(">>>set evaluate by GT in run %d>>>" % n_run)
		
		# out['overall']['set_meteor_hyp'], out['overall']['set_meteor_ref'], out['overall']['set_meteor_hau'], out['overall']['set_meteor_o2o'] = out_a, out_b, out_c, out_d
		
		logger.info(">>>set evaluate by GT in run %d>>>" % n_run)
		(out['overall']['set_meteor_hyp'], out['overall']['set_meteor_ref'], out['overall']['set_meteor_hau'], out['overall']['set_meteor_o2o']), set_all = eval_setAcc(json.load(open(annFile,'r'))['annotations'], pred_results, df_file_path=config['df_file_path'], metric="METEOR")

		for k,v in set_all.items():
			for kk,vv in v.items():
				out['ImgToEval'][k][kk] = vv
		
		# if not pred_path is None:
		# 	# add meteor hau as a metric.
		# 	for i in pred_results:
		# 		i['scores']['set_meteor_hau'] = out['ImgToEval'][i['image_id']]['hau']
		# 	json.dump(pred_results, open(pred_path[:-5]+'_gt_r%d.json'%n_run, "w"))
		
		# sys.exit(0)
		del out['overall']['oracle_image_id']
		del out['overall']['oracle_image_id_mean']
		# del out['overall']['oracle_Bleu_1']
		# del out['overall']['oracle_Bleu_2']
		# del out['overall']['oracle_Bleu_3']
		# del out['overall']['oracle_Bleu_1_mean']
		# del out['overall']['oracle_Bleu_2_mean']
		# del out['overall']['oracle_Bleu_3_mean']
		logger.info(
			'Output Metrics for accuracy by GT: %s',
			json.dumps(
				out['overall'],
				sort_keys=True,
				indent=4
			)
		)
		# sys.exit(0) # debug
		# if config.val_num_samples > 1 or config.val_group_size > 1:
		if True:
			logger.info(">>>We only use Psedo-GT and CR to evaluate Sampling")
			
			logger.info(">>>evaluate by Psedo-GT in run %d>>>" % n_run)
			annFile_p = config[mode+'_psedo_cocoformat']
			out_p = eval_oracle( annFile_p, pred_results, config['res_dir'], 'ours', 'val') # This line can't be omitted to change the scores in pred_results

			if not pred_path is None:
				json.dump(pred_results, open(pred_path[:-5]+'_pse_r%d.json'%n_run, "w"))
			'''
			rerank_pred_results = consensus_rerank(pred_results, topK=1)


			if not pred_rr_path is None:
				json.dump(rerank_pred_results, open(pred_rr_path[:-5]+"_top1_r%d.json"%n_run, "w"))

			logger.info(">>>Acc evaluation by GT after consensus reranking 1 in run %d>>>" % n_run)
			# out_acc_rerank = eval_oracle( annFile, rerank_pred_results, config['res_dir'], 'ours', 'val')
			'''
			out_acc_rerank = {}
			
			# out_acc_rerank['overall'] = eval_single(annFile, pred_rr_path[:-5]+"_top1_r%d.json"%n_run) # by the conventional one-caption coco evaluation.
			# logger.info(
			# 	'Output Metrics: for accuracy, after consensus reranking: %s',
			# 	json.dumps(
			# 		out_acc_rerank['overall'],
			# 		sort_keys=True,
			# 		indent=4
			# 	)
			# )
			
			# sys.exit(0)
			logger.info(">>>Div evaluation after concensus reranking %d from %d samples in run %d>>>" % 
						(config['div_topK'], int(config.val_num_samples*config.val_beam_width), n_run))
			rerank_pred_results_div = consensus_rerank(pred_results, topK=config['div_topK'])

			if not pred_rr_path is None:
				json.dump(rerank_pred_results_div, open(pred_rr_path[:-5]+"_top%d_r%d.json" % (config['div_topK'],n_run), "w"))
			print(config['pos_div_path'])
			# out_div_rerank = eval_div_stats(rerank_pred_results_div, 'ours', 'val', verb_path=config['pos_div_path'], df_file_path=config['df_file_path'])
			out_div_rerank = eval_div_stats(rerank_pred_results_div, 'ours', 'val', verb_path=config['coco_class'], df_file_path=config['df_file_path'])
			logger.info(
				'Output Metrics: for diversity, after consensus reranking: %s',
				json.dumps(
					out_div_rerank['overall'],
					sort_keys=True,
					indent=4
				)
			)
		
		
		else:
			logger.info("We use traditional lls to evaluate BS.")
			logger.info(">>>Acc evaluation by GT after ll score reranking 1 in run %d>>>" % n_run)
			if not pred_rr_path is None:
				rerank_lls_pred_results = pred_results[::snum]
				json.dump(rerank_lls_pred_results, open(pred_rr_path[:-5]+"_top1_lls%d.json"%n_run, "w"))
			out_acc_rerank = {}
			out_acc_rerank['overall'] = eval_single(annFile, pred_rr_path[:-5]+"_top1_lls%d.json"%n_run) # by the conventional one-caption coco evaluation.
			logger.info(
				'Output Metrics: for accuracy, after log likelihood reranking: %s',
				json.dumps(
					out_acc_rerank['overall'],
					sort_keys=True,
					indent=4
				)
			)

			logger.info(">>>Div evaluation after lls reranking %d from %d samples in run %d>>>" % 
						(config['div_topK'], snum, n_run))
			rerank_pred_results_div = []
			for ii in range(config['div_topK']):
				rerank_pred_results_div += pred_results[ii::snum]
			rerank_pred_results_div = sorted(rerank_pred_results_div, key=lambda jj: jj['image_id'])

			if not pred_rr_path is None:
				json.dump(rerank_pred_results_div, open(pred_rr_path[:-5]+"_top%d_r%d.json" % (config['div_topK'],n_run), "w"))

			out_div_rerank = eval_div_stats(rerank_pred_results_div, 'ours', 'val', verb_path=config['coco_class'], df_file_path=config['df_file_path'])

			logger.info(
				'Output Metrics: for diversity, after lls reranking: %s',
				json.dumps(
					out_div_rerank['overall'],
					sort_keys=True,
					indent=4
				)
			)

		logger.info("--- %s seconds for s%d-b%d-g%d ---" % (delta_time, config['val_num_samples'], config['val_beam_width'], config['val_group_size']))
		# out['overall']['all_time'] = delta_time

		out_all_runs.append(out)

		
		out_acc_rerank_all_runs.append(out_acc_rerank)
		out_div_rerank_all_runs.append(out_div_rerank)


		# return out, out_div_rerank, pred_results #,score_r1 #out_spice_n,
	return out_all_runs, out_acc_rerank_all_runs, out_div_rerank_all_runs

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default="msvd", type=str, help="The dataset that will evaluate.")
	# parser.add_argument('--ckpt_path', default="ckpts/pred_ckpts_msvd/model_checkpoint_0502_msvd_269.pt",type=str, help = "The path of the ckpt.")
	parser.add_argument('--res_dir', default='debug', 
									type=str, help='The root of pred file path.')
	parser.add_argument('--mode', default="val", type=str, choices = ["train","test","val"])
	parser.add_argument('--use_z', action="store_true", help="Whether to use latent variable z")
	parser.add_argument('--val_num_samples', required=True, type=int)
	parser.add_argument('--val_beam_width', default=1, type=int)
	parser.add_argument('--val_group_size', default=1, type=int)
	parser.add_argument('--dl', default=0.2, type=float)
	parser.add_argument('--debug', action='store_true', help='whether to debug.')
	parser.add_argument('--z_indeps', action="store_true", help="whether z2 is independent on z1.")
	parser.add_argument('--seq_CVAE', action="store_true", help="Whether uses seq_CVAE.")
	parser.add_argument('--nval', default=None, help="the sample number in the validation set")
	parser.add_argument('--pos_div_path', default=None, help="the path for diversity evaluation.")
	parser.add_argument('--div_topK', default=5, type=int, help="TopK captions used to evaluate the diversity.")
	parser.add_argument('--mul_runs', default=1, type=int, help="How many runs to pred the final results")
	parser.add_argument('--mark', default='-', help="Some marks when filing results")
	parser.add_argument('--rand_seed', default=1234, type=int)
	parser.add_argument('--scale_o', default=0.35, type=float, help="The scale of object variance.")
	parser.add_argument('--use_pos', action='store_true', help='whether to use_pos.')
	parser.add_argument('--paraAug', action="store_true", help='whether augment sentences using paraphrases')
	parser.add_argument('--external_file', default=None, type=str, help='The short cut of external prediction file.')
 	
	args = parser.parse_args()
	random.seed(args.rand_seed)
	np.random.seed(args.rand_seed)
	torch.manual_seed(args.rand_seed)

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
	else:
		logger.error("Dataset Name Error!")
	config = json.loads(open(config_file, 'r').read())
	if dname == "mid_vatex":
		config['train_cocoformat'] = "datasets/vatex/mid_vatex_train_cocofmt_0th.json"
	elif dname == "mini_vatex":
		config['train_cocoformat'] = "datasets/vatex/mini_vatex_train_cocofmt_0th.json"
	config = config['params']
	config['rand_seed'] = args.rand_seed
	data_path = config['pathToData']
	vocab_path = config['vocab_path']
	coco_class = config['coco_class']
	img_data_path = config['image_data_path']
	val_batch_size = 100 if args.debug else int(config['val_batch_size']) # here batch_size of train and val are the same!
	config['batch_size'] = val_batch_size
	nseqs_per_video = int(config['nseqs_per_video'])
	feat_inputs_name = config['visionfeats']+"_z" if args.use_z else config['visionfeats']
	vocab_size = config['vocab_size']
	config['use_z'] = args.use_z
	config['val_beam_width'] = args.val_beam_width
	config['val_group_size'] = args.val_group_size
	config['diversity_lambda'] = args.dl
	config['val_num_samples'] = args.val_num_samples
	config['res_dir'] = args.res_dir
	config['z_indeps'] = args.z_indeps
	config['seq_CVAE'] = args.seq_CVAE
	config['div_topK'] = args.div_topK
	config['scale_o'] = args.scale_o
	mode = args.mode
	if not args.pos_div_path is None:
		config['pos_div_path'] = args.pos_div_path
	else:
		config['pos_div_path'] = coco_class

	ckpt_path = os.path.join(args.res_dir, "best_model.ckpt")
	if not os.path.exists(os.path.join(args.res_dir, args.mode)):
	# 	shutil.rmtree(os.path.join(args.res_dir, args.mode))
		os.mkdir(os.path.join(args.res_dir, args.mode))
	
	if not os.path.exists(os.path.join(args.res_dir, 'scripts')):
	# 	shutil.rmtree(os.path.join(args.res_dir, 'scripts'))
		os.mkdir(os.path.join(args.res_dir, 'scripts'))

	shutil.copy('scripts/train.py', os.path.join(args.res_dir, 'scripts/train.py'))
	shutil.copy('scripts/eval.py', os.path.join(args.res_dir, 'scripts/eval.py'))
	shutil.copy('modules/textmodules.py', os.path.join(args.res_dir, 'scripts/textmodules.py'))

	global_result_path = "datasets/%s_%s_gresult.csv" % (args.dataset, args.mode)

	# CHANGE LATER!!!
	pred_path = os.path.join(args.res_dir, args.mode, "%s_s%d_b%d_g%d_%s_%s_ori" % (mode, config['val_num_samples'], config['val_beam_width'], config['val_group_size'], str(config['diversity_lambda']), args.mark) + ".json")
	log_path = os.path.join(args.res_dir, args.mode, "%s_s%d_b%d_g%d_%s_%s_ori" % (mode, config['val_num_samples'], config['val_beam_width'], config['val_group_size'], str(config['diversity_lambda']), args.mark) + ".log")
	pred_rr_path = os.path.join(args.res_dir, args.mode, "%s_s%d_b%d_g%d_%s_%s_rerank" % (mode, config['val_num_samples'], config['val_beam_width'], config['val_group_size'], str(config['diversity_lambda']), args.mark) + ".json")
	if os.path.isfile(log_path):
		os.remove(log_path)

	logger.info(log_path)
	logging.basicConfig(
		level=getattr(logging, "DEBUG"),
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(log_path), # change later.
			logging.StreamHandler()
			]
	)

	logger.info("normal sampling, sampling number: %s." % str(args.val_num_samples))

	logger.info(
		'Input arguments: %s',
		json.dumps(
			config,
			sort_keys=True,
			indent=4
		)
	)

	test_dataset = COCO_Dataset(data_path,img_data_path,vocab_path,coco_class,
								nseqs_per_video=nseqs_per_video,mode=mode,nval=args.nval, dataset_name=args.dataset)
	logger.info("len of test set: "+str(len(test_dataset)))
	val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], 
								shuffle=False, num_workers=1, drop_last=True, collate_fn=my_collate)

	logger.info("Load the ckpt file from %s" % ckpt_path)
	try:
		with open(ckpt_path, "rb") as f:
			best_ckpt = torch.load(f)
	except Exception:
		print("ckpt not found!")
		# best_ckpt = None
		sys.exit(0)

	config = DotMap(config)
	pos_path = args.res_dir+'/sample_z_pos_pred_Train.h5' if args.use_pos else None
	out_all, out_sinacc_all, out_div_all= eval(val_dataloader, best_ckpt, config, logger, write_result=True, mode=mode, 
				debug=args.debug, pred_path=pred_path, pred_rr_path=pred_rr_path, mul_runs=args.mul_runs, use_pos=pos_path, paraAug=args.paraAug, external_file=args.external_file)
	logger.info("The ckpt file is from %s" % ckpt_path)
	logger.info("The val log file is saved to %s" % log_path)
	logger.info("The prediction file is saved to %s" % pred_path)

	'''
	for n_run in range(args.mul_runs):
		append2csv(config, logger, out_all[n_run]['overall'],out_sinacc_all[n_run]['overall'], out_div_all[n_run]['overall'], args.mark)

	def avg(all_res):
		X = np.array([[v for k,v in all_res[i]['overall'].items()] for i in range(len(all_res))])
		keys = [k for k,_ in all_res[0]['overall'].items()]
		my_results = {k:v for k, v in zip(keys, np.mean(X,axis=0))}
		return my_results

	def std(all_res):
		X = np.array([[v for k,v in all_res[i]['overall'].items()] for i in range(len(all_res))])
		keys = [k for k,_ in all_res[0]['overall'].items()]
		my_results = {k:v for k, v in zip(keys, np.std(X,axis=0))}
		return my_results

	append2csv(config, logger, avg(out_all),avg(out_sinacc_all), avg(out_div_all), 'AVG')
	append2csv(config, logger, std(out_all),std(out_sinacc_all), std(out_div_all), 'STD')
	'''

