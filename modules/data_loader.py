from __future__ import print_function, division
import json
import h5py
import numpy as np
import sys
import os
import csv
import torch
import base64
import copy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import pickle
sys.path.insert(0,'./data')
from build_vocab_coco import Vocabulary
import nltk
from nltk.tokenize import word_tokenize as tokenize
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import inflect
import random

inflect = inflect.engine() # for handling plural forms in the captions for pseudo-supervision
csv.field_size_limit(sys.maxsize)

import logging
logger = logging.getLogger(__name__)

class COCO_Dataset(Dataset):
	"""COCO dataset."""

	def __init__(self, text_data_path,image_data_path, vocab_path,coco_class,mode="test",maxlength=28, maxlength_obj=28, nseqs_per_video=17,nval=None,dataset_name="msvd", ngram=1, result_dir = None):
		"""
		Args:
			text_data_path (string list): Path to the json file with captions or annotations.
			image_data_path (string): tsv file with image features
			vocab_path (string): Path to the vocab pickle file.
			coco_class (coco_class): list of coco classes removed from captions to get contexual descriptions
			mode: "train" / "test" / "val"
			ntest: additional reference to the sample number in the validation set.
		"""
		"""
		Returns:
		captions: The ground-truth captions 
		bottom_up_features: Features from bounding boxes extracted from Faster-RCNN [4]
		bottom_up_classes: Classes from bounding boxes corresponding to bottom_up_features
		x_m_caps: Contextual descriptions after removing COCO classes from captions
		caption_length: Caption lengths
		x_o_caps: Object descriptions of COCO classes from captions
		video_idx: image-id  in the annotations file

		"""

		self.mode = mode
		self.nval = nval
		self.ngram = ngram
		self.image_data_path = image_data_path # list
		self.vocab = json.load(open(str(vocab_path), "r"))
		self.word2idx = {w: i for i, w in enumerate(self.vocab)}
		self.idx2word = {i: w for i, w in enumerate(self.vocab)}
		self.maxlength = maxlength
		self.maxlength_obj = maxlength_obj
		self.nseqs_per_video = nseqs_per_video
		self.result_dir = result_dir
		coco_class_all = []
		coco_class_name = open(coco_class, 'r')
		if coco_class[-3:] == "txt":
			for line in coco_class_name:
				coco_class = line.rstrip("\n").split(', ')
				coco_class_all.append(coco_class)
		elif coco_class[-4:] == "json":
			coco_class_all = json.load(coco_class_name)
			coco_class_all = [[i] for i in coco_class_all]
		self.wtod = {}
		for i in range(len(coco_class_all)):
			for w in coco_class_all[i]:
				self.wtod[w] = i
		# self.hit_set = [WordNetLemmatizer().lemmatize(w,'v') for w in self.wtod]
		self.wtol = {} # word to lemmatizer
		lemmatizer = WordNetLemmatizer()
		for w in self.word2idx:
			if len(tokenize(w)) == 0:
				print(w)
			tok = tokenize(w)[0]
			# self.wtol[w] = lemmatizer.lemmatize(tok, 'v') # For verb.
			self.wtol[w] = tok
		self.dtoi = {w:i+1 for i,w in enumerate(self.wtod.keys())}

		self.imagefeatures  = {}
		if dataset_name == "coco":
			logger.info('DataLoader loading h5 file: %s', self.image_data_path[0])
			self.imagefeatures_train_h5 = h5py.File(self.image_data_path[0],'r')
			logger.info('DataLoader loading h5 file: %s', self.image_data_path[1])
			self.imagefeatures_val_h5 = h5py.File(self.image_data_path[1], 'r')
		else: # for msvd, msrvtt, vatex (change for different experiments)
			logger.info('DataLoader loading h5 file: %s', self.image_data_path[0])
			self.imagefeatures_23D_h5 = h5py.File(open(self.image_data_path[0], 'rb'), 'r')
			# self.imagefeatures_r_h5 = h5py.File(open(self.image_data_path[1], 'rb'), 'r')
		self.dname = dataset_name
		if dataset_name == "msvd":
			self.splits = [1200,100,670]
		elif dataset_name == "msrvtt":
			self.splits = [6513,497,2990] # msrvtt
		elif dataset_name == "vatex": # vatex
			# self.splits = [25991,3000,6000] # original validation set
			if self.nval is None:
				self.splits = [25991, 500, 6000]
			else:
				self.splits = [25991, int(self.nval), 6000]
		elif dataset_name == "coco": # coco (image)
			self.splits = [118287, 500, 1000]
		elif dataset_name == "mini_vatex":
			self.splits = [25991, int(self.nval), 6000]#[1200, 100, 670], change later
		elif dataset_name == "mid_vatex":
			self.splits = [25991, int(self.nval), 6000]#[6513,497,2990]

		self.n_videos = int(sum(self.splits)) # useless for coco

		# for k in self.imagefeatures_h5.keys():
		# 	self.imagefeatures[k] = self.imagefeatures_h5[k][()]#.astype(np.float32)

		'''Things are different for COCO dataset, for the index is given for train, test and val
		   , Not just in an order.
		'''
		if dataset_name == "coco":
			val_id_list = open('datasets/coco/coco_val_mRNN.txt','r').read().split('\n')
			test_id_list = open('datasets/coco/coco_test_mRNN.txt','r').read().split('\n')
			val_id_list = [int(i[13:-4]) for i in val_id_list]
			test_id_list = [int(i[13:-4]) for i in test_id_list]
			id2inx_valset = {cont:i for i, cont in enumerate(self.imagefeatures_val_h5['image_id'])}
			val_inx_list = sorted([id2inx_valset[i] for i in val_id_list])
			test_inx_list = sorted([id2inx_valset[i] for i in test_id_list])

			if self.mode == "train":
				self.imagefeatures['feats'] = np.delete(self.imagefeatures_val_h5['features'], val_inx_list+test_inx_list, axis=0)
				self.imagefeatures['video_id'] = np.delete(self.imagefeatures_val_h5['image_id'], val_inx_list+test_inx_list, axis=0)
				self.imagefeatures['numframes'] = np.delete(self.imagefeatures_val_h5['num_boxes'], val_inx_list+test_inx_list, axis=0)

				self.imagefeatures['feats'] = np.hstack([self.imagefeatures_train_h5['features'], self.imagefeatures['feats']])
				self.imagefeatures['video_id'] = np.hstack([self.imagefeatures_train_h5['image_id'], self.imagefeatures['video_id']])
				self.imagefeatures['numframes'] = np.hstack([self.imagefeatures_train_h5['num_boxes'], self.imagefeatures['numframes']])

				assert len(self.imagefeatures['feats']) == len(self.imagefeatures['video_id']) \
					== len(self.imagefeatures['numframes']) == self.splits[0], "Sir, the length of trainset should be 118287!"
				
			elif self.mode == "val":
				self.imagefeatures['feats'] = self.imagefeatures_val_h5['features'][val_inx_list[:500]] # shrink the valset
				self.imagefeatures['video_id'] = self.imagefeatures_val_h5['image_id'][val_inx_list[:500]]
				self.imagefeatures['numframes'] = self.imagefeatures_val_h5['num_boxes'][val_inx_list[:500]]

				# assert len(self.imagefeatures['feats']) == len(self.imagefeatures['video_id']) \
				# 	== len(self.imagefeatures['numframes']) == self.splits[1], "Sir, the length of valset should be 4000!"

			else:
				self.imagefeatures['feats'] = self.imagefeatures_val_h5['features'][test_inx_list]
				self.imagefeatures['video_id'] = self.imagefeatures_val_h5['image_id'][test_inx_list]
				self.imagefeatures['numframes'] = self.imagefeatures_val_h5['num_boxes'][test_inx_list]

				assert len(self.imagefeatures['feats']) == len(self.imagefeatures['video_id']) \
					== len(self.imagefeatures['numframes']) == self.splits[2], "Sir, the length of trainset should be 1000!"

		else: # for datasets except for COCO.
			video_ids = [i for i in range(self.n_videos)]
			if self.mode == "train":
				for k in self.imagefeatures_23D_h5.keys(): # HDF5 dataset "feats": shape (1970, 26, 2560), type "<f4"
					self.imagefeatures[k] = self.imagefeatures_23D_h5[k][:self.splits[0]] # for MSVD
				# for k in self.imagefeatures_r_h5.keys(): # HDF5 dataset "sfeats": shape (1970, 26, 36, 5), type "<f4"
				# 										# HDF5 dataset "vfeats": shape (1970, 26, 36, 2048), type "<f4"
				# 	self.imagefeatures[k] = self.imagefeatures_r_h5[k][:self.splits[0]]
				self.imagefeatures['video_id'] = video_ids[:self.splits[0]]

			elif self.mode == "val":
				for k in self.imagefeatures_23D_h5.keys(): # HDF5 dataset "feats": shape (1970, 26, 2560), type "<f4"
					self.imagefeatures[k] = self.imagefeatures_23D_h5[k][self.splits[0]:self.splits[0]+self.splits[1]] # for MSVD
				# for k in self.imagefeatures_r_h5.keys(): # HDF5 dataset "sfeats": shape (1970, 26, 36, 5), type "<f4"
				# 										# HDF5 dataset "vfeats": shape (1970, 26, 36, 2048), type "<f4"
				# 	self.imagefeatures[k] = self.imagefeatures_r_h5[k][self.splits[0]:self.splits[0]+self.splits[1]]
				self.imagefeatures['video_id'] = video_ids[self.splits[0]:self.splits[0]+self.splits[1]]
				
			else:
				for k in self.imagefeatures_23D_h5.keys(): # HDF5 dataset "feats": shape (1970, 26, 2560), type "<f4"
					self.imagefeatures[k] = self.imagefeatures_23D_h5[k][self.splits[0]+self.splits[1]:self.n_videos] # for MSVD
				# for k in self.imagefeatures_r_h5.keys(): # HDF5 dataset "sfeats": shape (1970, 26, 36, 5), type "<f4"
				# 										# HDF5 dataset "vfeats": shape (1970, 26, 36, 2048), type "<f4"
				# 	self.imagefeatures[k] = self.imagefeatures_r_h5[k][self.splits[0]+self.splits[1]:self.n_videos]
				self.imagefeatures['video_id'] = video_ids[self.splits[0]+self.splits[1]:self.n_videos]
	

		self.inv_annotations = {}
		self.cid = {} # cluster_id

		logger.info('DataLoader loading annotation json file: %s', text_data_path[0])
		annotations = json.load(open(text_data_path[0]))['annotations']
		# if self.dname == "coco":
		# 	logger.info('DataLoader loading annotation json file: %s', text_data_path[1])
		# 	annotations += json.load(open(text_data_path[1]))['annotations']
		# assert the annotation file, {'annotations': [{caption: str, image_id: int}...]}

		for c in annotations:
			if c["image_id"] in self.inv_annotations.keys(): # from 0th
				self.inv_annotations[c["image_id"]].append(c["caption"])
			else:
				self.inv_annotations[c["image_id"]] = []
				self.inv_annotations[c["image_id"]].append(c["caption"])
		# cid records the cluster id for each caption
		for c in annotations:
			if c["image_id"] in self.cid.keys(): # from 0th
				self.cid[c["image_id"]].append(c["cluster_id"])
			else:
				self.cid[c["image_id"]] = []
				self.cid[c["image_id"]].append(c["cluster_id"])
		
		# for c in annotations["images"]:
		# 	if str(c["id"]) not in self.image_filenames.keys():
		# 		self.image_filenames[str(c["id"])] = c["file_name"]
		
		self.blacklist_classes = {
				"auto part":'vehicle', "bathroom accessory":'furniture', "bicycle wheel":'bicycle', "boy":'boy',
				"door handle":'door', "fashion accessory":'clothing', "footwear":'shoes', "human arm":'person',
				"human beard":'person', "human body":'person', "human ear":'person', "human eye":'person', "human face":'person', "human foot":'person',
				"human hair":'person', "human hand":'person', "human head":'person', "human leg":'person', "human mouth":'person', "human nose":'person',
				"land vehicle":'vehicle', "plumbing fixture":'toilet',
				"seat belt":'vehicle', "vehicle registration plate":'vehicle',
				"face":'person',"hair":'person',"head":'person',"ear":'person',"tail":'giraffe',"neck":'giraffe',
				"hat":'person',"helmet":'person',"nose":'person',"tire":'bus',"tour":'bus',"hand":'person',"shadow":'person'
			}

		self.punctuations = [
			"''", "'", "``", "`", "(", ")", "{", "}",
			".", "?", "!", ",", ":", "-", "--", "...", ";"
		]

		# self.vg_classes_to_vocab = {}
		# self.vg_classes_to_vocab[0] = 0
		# self.vg_classes_to_vocab_p = {}
		# self.vg_classes_to_vocab_p[0] = 0
		# classes = ['__background__']
		# vg_obj_counter = 1
		# with open('./data/visual_genome_classes.txt') as f:
		# 	for _object in f.readlines():
		# 		#classes.append(object.split(',')[0].lower().strip())
		# 		_object = _object.split(',')[0].lower().strip()
		# 		if _object in self.word2idx:
		# 			if _object in self.blacklist_classes:
		# 				self.vg_classes_to_vocab[vg_obj_counter] = self.word2idx[self.blacklist_classes[_object]]
		# 				self.vg_classes_to_vocab_p[vg_obj_counter] = self.word2idx[self.blacklist_classes[_object]]
		# 			else:
		# 				self.vg_classes_to_vocab[vg_obj_counter] = self.word2idx[_object]
		# 				if inflect.singular_noun( _object ) == False:
		# 					_object_p = inflect.plural(_object)
		# 				else:
		# 					_object_p = _object
		# 				if _object_p in self.word2idx:
		# 					self.vg_classes_to_vocab_p[vg_obj_counter] = self.word2idx[_object_p]
		# 				else:
		# 					self.vg_classes_to_vocab_p[vg_obj_counter] = self.word2idx[_object]
		# 		else:
		# 			self.vg_classes_to_vocab[vg_obj_counter] = 0
		# 			self.vg_classes_to_vocab_p[vg_obj_counter] = 0
		# 		vg_obj_counter += 1

		# if not result_dir is None:
		# 	self.get_compared_caps()

	def get_det_word(self,captions, ngram=1):
		# get the present category. taken from NBT []
		indicator = []
		stem_caption = []
		for s in captions:
			tmp = []
			for w in s:
				# for verb it needs to be present tense
				if w in self.wtol.keys():	
					tmp.append(self.wtol[w])
			stem_caption.append(tmp)
			indicator.append([(-1, -1, -1)]*len(s)) # category class, binary class, fine-grain class.

		ngram_indicator = {i+1:copy.deepcopy(indicator) for i in range(ngram)}
		# get the 2 gram of the caption.
		
		for i, s in enumerate(stem_caption):
			for n in range(ngram,0,-1):
				#print('stem_caption ', s)
				for j in range(len(s)-n+1):
					ng = ' '.join(s[j:j+n])
					#print('ng ', ng)
					# if the n-gram exist in word_to_detection dictionary.
					if ng in self.wtod and indicator[i][j][0] == -1: #and self.wtod[ng] in pcats: # make sure that larger gram not overwright with lower gram.
						bn = (ng != ' '.join(captions[i][j:j+n])) + 1
						fg = self.dtoi[ng]
						#print('fg ',fg)
						ngram_indicator[n][i][j] = (int(self.wtod[ng]), int(bn), int(fg))
						indicator[i][j:j+n] = [(int(self.wtod[ng]), int(bn), int(fg))] * n
			#sys.exit(0)
		return ngram_indicator


	def get_caption_seq(self,captions,cap_len,nseqs_per_video=5, ngram=2):
		# NOTE: here, the length of object_cap_seq DOES not indicate the seq_len but the number of possible objects
		# assert nseqs_per_video <= len(cap_len), "nseq_per_video should less than all captions"
		# To increase more paired data, try to use all paired data.
		# cap_seq = np.zeros([nseqs_per_video, self.maxlength], dtype=int)
		# masked_cap_seq = np.zeros([nseqs_per_video, self.maxlength], dtype=int)
		# object_cap_seq = np.zeros([nseqs_per_video, self.maxlength], dtype=int)
		# cap_seq_all = np.zeros((len(captions), self.maxlength), dtype=int)
		def getuni_inx(mylist):
			inds = []
			unseen = set()
			mylist = [" ".join(i[1:-1]) for i in mylist]
			for ix, item in enumerate(mylist):
				if item not in unseen:
					inds.append(ix)
				unseen.add(item)
			return inds

		# Remove the repeated captions.
		# uni_ix = getuni_inx(captions)
		# captions = [captions[i] for i in uni_ix]
		# cap_len = [cap_len[i] for i in uni_ix]

		cap_seq = np.zeros([len(cap_len), self.maxlength], dtype=int)
		masked_cap_seq = np.zeros([len(cap_len), self.maxlength], dtype=int)
		# object_cap_seq = np.zeros([len(cap_len), self.maxlength], dtype=int)
		object_cap_seq = np.zeros([len(cap_len), self.maxlength_obj], dtype=int)

		det_indicator = self.get_det_word(captions, ngram)
		# original one
		# for i, caption in enumerate(captions):
		# 	j = 0
		# 	k = 0
		# 	o = 0
		# 	while j < len(caption) and j < self.maxlength:
		# 		is_det = False
		# 		for n in range(ngram, 0, -1):
		# 			if det_indicator[n][i][j][0] != -1:
		# 				cap_seq[i,k] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
		# 				# if inflect.singular_noun( caption[j] ) == False:
		# 				# 	masked_cap_seq[i,k] = 4 #placeholder in vocab for singular visual genome class object
		# 				# else:
		# 				# 	masked_cap_seq[i,k] = 3 #placeholder in vocab for plural visual genome class object
		# 				masked_cap_seq[i,k] = 4 # for verb
		# 				object_cap_seq[i,o] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
		# 				is_det = True
		# 				j += n # skip the ngram.
		# 				o += 1
		# 				break
		# 		if is_det == False:
		# 			cap_seq[i,k] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
		# 			masked_cap_seq[i,k] = cap_seq[i,k]
		# 			j += 1
		# 		k += 1

		# # [WARNING] DEBUDING, DELETE LATER
		# cap_seq = np.zeros([len(cap_len), self.maxlength], dtype=int)
		# masked_cap_seq = np.zeros([len(cap_len), self.maxlength], dtype=int)
		# # object_cap_seq = np.zeros([len(cap_len), self.maxlength], dtype=int)
		# object_cap_seq = np.zeros([len(cap_len), self.maxlength_obj], dtype=int)

		# Exp 2: a new plan to remove the verb, just matching.
		# [UPDATING]: to use the verb extractor to get the verbs 
		# for i in range(len(captions)):
		# 	pos = nltk.pos_tag(captions[i])
		# 	jj = 0
		# 	for j in range(self.maxlength):
		# 		cap_seq[i,j] = int(self.word2idx[captions[i][j]] if captions[i][j] in self.word2idx.keys() else 3) # <unk>
		# 		if cap_seq[i,j] > 5 and pos[j][1][:2] == "VB": 
		# 			object_cap_seq[i,jj] = cap_seq[i,j]
		# 			jj += 1
		# 		else:
		# 			masked_cap_seq[i,j] = cap_seq[i,j]
		# 		if cap_seq[i,j] == 2: # <end>
		# 			break

		# Exp 3: let the shortest caption as the object_cap_seq, other sentences as masked_cap_seq
		# sorted_ix = np.argsort(cap_len)
		# shorter_ix = [sorted_ix[0]] * nseqs_per_video
		# longer_ix = sorted_ix[-(nseqs_per_video):]
		# for i in range(len(captions)):
		# 	for j in range(self.maxlength):
		# 		cap_seq_all[i,j] = int(self.word2idx[captions[i][j]] if captions[i][j] in self.word2idx.keys() else 3)
		# 		if cap_seq_all[i,j] == 2: # <end>
		# 			break
		# masked_cap_seq = cap_seq_all[shorter_ix]
		# object_cap_seq = cap_seq_all[longer_ix]
		# cap_seq = object_cap_seq
		# cap_len = [cap_len[i] for i in longer_ix]

		# Exp 4: let all the verbs in a caption set be extracted to form a "verb" space.
		# Test: 
		locAllVerbs = []

		for i, caption in enumerate(captions):
			j = 0
			k = 0
			o = 0
			while j < len(caption) and j < self.maxlength:
				is_det = False
				for n in range(ngram, 0, -1):
					if det_indicator[n][i][j][0] != -1:
						cap_seq[i,k] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
						# if inflect.singular_noun( caption[j] ) == False:
						# 	masked_cap_seq[i,k] = 4 #placeholder in vocab for singular visual genome class object
						# else:
						# 	masked_cap_seq[i,k] = 3 #placeholder in vocab for plural visual genome class object
						masked_cap_seq[i,k] = 4 # for verb
						# object_cap_seq[i,o] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
						locAllVerbs.append(int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0))
						is_det = True
						j += n # skip the ngram.
						o += 1
						break
				if is_det == False:
					cap_seq[i,k] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
					masked_cap_seq[i,k] = cap_seq[i,k]
					j += 1
				k += 1
		locAllVerbs = list(set(locAllVerbs))

		# self.maxlength is not for the number of objects
		object_cap_seq[:, :len(locAllVerbs)] = locAllVerbs[:self.maxlength_obj]

		return cap_seq, masked_cap_seq, object_cap_seq, cap_len


	def __len__(self):
		return len(self.imagefeatures["feats"])
		# return len(self.imagefeatures["vfeats"])

	def get_i2w(self):
		return self.idx2word

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		captions = []
		masked_images = []
		x_m_caps = []
		x_o_caps = []
		video_idx = self.imagefeatures["video_id"][idx]
		# if str(video_idx) in self.image_filenames.keys():  
		# 	i=idx
		# else:
		# 	#
		# 	video_idx = self.imagefeatures["video_id"][1]
		# 	i=1
		
		# (26 * 2560+2048=4608) it is like 26 "objects", (which is actually "timestep.")
		# and 2560 = 1536 (IRV2)+1024 (I3D) (Note the order!) 
		# (26, 36, 2048) to (26, 2048)
		'''
		Exp01_1: to take [averaged objects in each frame ,2D ,3D] as visual features.
		Exp01_2: to take [2D ,3D] as visual features.
		Exp01_3: to take [middle objects in each frame] as visual features.
		Exp01_4: to take [3D] as visual features

		'''
		if self.dname in ["msvd", "msrvtt"]:
			# #Exp01_1
			# bottom_up_features = np.mean(np.array(self.imagefeatures['vfeats'][idx]), axis=1) # average by object dims.
			# bottom_up_features = np.concatenate([np.array(self.imagefeatures['sfeats'][idx]), bottom_up_features], axis=1) # ()
			# Exp01_2
			bottom_up_features = np.array(self.imagefeatures['feats'][idx])
			#Exp01_3
			# bottom_up_features = np.array(self.imagefeatures['vfeats'][idx])
			# bottom_up_features = bottom_up_features[bottom_up_features.shape[0]//2] 

			# #Exp01_4
			# bottom_up_features = np.array(self.imagefeatures['feats'][idx])[:,-1024:]

		#Exp02: to take only [3D] as visual features.
		else: # for vatex and coco
			bottom_up_features = []
			# bottom_up_classes = []
			# bottom_up_classes_p = []
			total_boxes = 0
			nms_boxes = []
			nms_class_names = []
			imagefeature_size = int(self.imagefeatures['feats'][0].shape[0]/self.imagefeatures['numframes'][0])
			i = idx
			for j in range(self.imagefeatures["numframes"][i]):
				bottom_up_features.append( self.imagefeatures["feats"][i][(j)*imagefeature_size:((j)+1)*imagefeature_size] )
				# bottom_up_classes.append(self.vg_classes_to_vocab[ self.imagefeatures["classes"][i][j] ])
				# bottom_up_classes_p.append(self.vg_classes_to_vocab_p[ self.imagefeatures["classes"][i][j] ])
			bottom_up_features = np.array(bottom_up_features)
			# bottom_up_classes = np.array(bottom_up_classes)
			# bottom_up_classes_p = np.array(bottom_up_classes_p)
			if bottom_up_features.shape[0] < 100:
				bottom_up_features_pad = np.zeros((100 - bottom_up_features.shape[0], imagefeature_size))
				# bottom_up_classes_pad = np.zeros((100 - bottom_up_classes.shape[0],))
				bottom_up_features = np.concatenate([bottom_up_features,bottom_up_features_pad],axis=0) 
				# bottom_up_classes = np.concatenate([bottom_up_classes,bottom_up_classes_pad],axis=0)
				# bottom_up_classes_p = np.concatenate([bottom_up_classes_p,bottom_up_classes_pad],axis=0)
			# bottom_up_classes = np.concatenate([bottom_up_classes,bottom_up_classes_p], axis=0)

		bottom_up_classes = None # no class information for now.
		caps = self.inv_annotations[video_idx] ## here should be the "video_idx" not the systemetic index "idx".
		cids = np.array([ int(i) for i in self.cid[video_idx] ])
		targets = []
		caption_length = []
		for c in caps:
			caption_tokens = nltk.tokenize.word_tokenize(c.lower().strip())
			caption_tokens = [ct for ct in caption_tokens if ct not in self.punctuations]
			caption = []
			caption.append('<start>')
			caption.extend(caption_tokens)
			caption = caption[0:(self.maxlength-1)]
			caption.append('<end>')
			targets.append(caption)
			caption_length.append(len(caption))

		gt_cap, mask_cap, obj_cap, caption_length = self.get_caption_seq(targets, caption_length, self.nseqs_per_video, ngram=self.ngram) #5 human annotations

		info_dict = {'video_idx': video_idx,
					 'caption_sent': self.arr2lang(gt_cap),
					 'cap_mask_sent': self.arr2lang(mask_cap),
					 'cap_obj_sent': self.arr2lang(obj_cap)}
		return gt_cap, mask_cap, obj_cap, cids, bottom_up_features, np.array(caption_length), video_idx, info_dict,  

	def get_compared_caps(self):
		'''
		To get the triplet captions, (original, only verb captions, context captions)
		'''
		vids = []
		gt_caps = []
		mask_caps = []
		obj_caps = []

		if self.mode == "train":
			all_n = self.splits[0]
		elif self.mode == "val":
			all_n = self.splits[1]
		else:
			all_n = self.splits[2]

		for vid in range(all_n):
			video_idx = self.imagefeatures["video_id"][vid]
			caps = self.inv_annotations[video_idx] ## here should be the "video_idx" not the systemetic index "idx".
			targets = []
			caption_length = []
			for c in caps:
				caption_tokens = nltk.tokenize.word_tokenize(c.lower().strip())
				caption_tokens = [ct for ct in caption_tokens if ct not in self.punctuations]
				caption = []
				caption.extend(caption_tokens)
				caption = caption[0:(self.maxlength-1)]
				caption.append('<end>')
				targets.append(caption)
				caption_length.append(len(caption))

			gt_cap, mask_cap, obj_cap, caption_length = self.get_caption_seq(targets, caption_length, self.nseqs_per_video, ngram=self.ngram)
			gt_cap_list = self.arr2lang(gt_cap)
			mask_cap_list = self.arr2lang(mask_cap)
			obj_cap_list = self.arr2lang(obj_cap)

			vids += [vid] * len(gt_cap)
			gt_caps += gt_cap_list
			mask_caps += mask_cap_list
			obj_caps += obj_cap_list

		df = pd.DataFrame()
		df['video_ids'] = vids 
		df['gt_captions'] = gt_caps
		df['mask_captions'] = mask_caps
		df['obj_captions'] = obj_caps

		save_path = self.result_dir + "/caption_summary_%s.csv" % self.mode
		df.to_csv(save_path, index=False)
		logger.debug("The summary of captions, masks, and objs are saved into %s" % save_path)

	def arr2lang(self, arr):
		T, N = arr.shape
		return [" ".join([self.get_i2w()[arr[t,n]] for n in range(N) if arr[t,n]>4]) for t in range(T)]#if arr[t,n]>4

def global_shuffle(dataloader, bs):
	# A coco-caption like dataloader
	# return an iterator, Like a list: [[new_batch1], [new_batch2], [new_batch...]]
	# The aim is to shuffle the dataloader *globally*: make the captions for the same vision not appearing in one batch

	results ={i:[] for i in range(9)}
	for data in dataloader:
		seq_len_per_video = [len(i) for i in data[0]]
		results[0] += seq_len_per_video # seq_len_per_video, useless
		results[1].append(torch.cat(data[0])) # caps
		results[2].append(torch.cat(data[1])) # mask_cap
		results[3].append(torch.cat(data[2])) # obj_caps
		results[4].append(torch.cat(data[3])) # cid

		results[5].append(torch.cat([d.unsqueeze(0) for ix,d in enumerate(data[4]) for j in range(seq_len_per_video[ix])])) # bfeats
		results[6].append(torch.cat(data[5])) # seq_lengths
		results[7] += [d for ix,d in enumerate(data[6]) for j in range(seq_len_per_video[ix])]
		results[8] += [d for ix,d in enumerate(data[7]) for j in range(seq_len_per_video[ix])] # to change some, cap_info: [WARNING] not align
	
	for i in range(1,7):
		results[i] = torch.cat(results[i])
	
	all = list(zip(results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8]))
	random.shuffle(all)

	s = list(range(0, len(all), bs))

	final = [tuple((zip(*all[s[i]:s[i]+bs]))) for i in range(len(s))]
	def convt(final, i, j):
		if j < 6:
			return torch.stack(final[i][j])
		else:
			return np.array(final[i][j])

	final_cp = [[convt(final, i, j) for j in range(8) ] for i in range(len(final)) ]
		       

	return final_cp

def my_collate(batch):
	# input: [(captions, x_m_caps, x_o_caps, bottom_up_features, np.array(caption_length), video_idx, info_dict), ..., ()]
	caps = [torch.LongTensor(i[0]) for i in batch]
	x_m_caps = [torch.LongTensor(i[1]) for i in batch]
	x_o_caps = [torch.LongTensor(i[2]) for i in batch]
	cids = [torch.LongTensor(i[3]) for i in batch]

	bu_feats = [torch.FloatTensor(i[4]) for i in batch]
	cap_lens = [torch.LongTensor(i[5]) for i in batch]
	video_idx = [i[6] for i in batch]
	info_dict = [i[7] for i in batch]

	return [caps, x_m_caps, x_o_caps, cids, bu_feats, cap_lens, video_idx, info_dict]

if __name__ == "__main__":
	data_path = ["datasets/msvd/annotations_msvd.json"]
	vocab_path = "datasets/msvd/msvd_vocab.json"
	# coco_class = "datasets/msvd/verb_class_msvd.txt"
	coco_class = "datasets/msvd/msvd_train_verb_g4.json"
	img_data_path = ["datasets/msvd/msvd_features.h5"]
	n_captions_per_image = 17
	# data_path = ["./datasets/coco/annotations/captions_train2014.json", "./datasets/coco/annotations/captions_val2014.json"]
	# vocab_path = "datasets/coco/coco_vocab.json"
	# coco_class = "data/coco_classname.txt"
	# # coco_class = "data/verb_class.txt"
	# img_data_path = ["./datasets/coco/coco_train_2014_adaptive_withclasses.h5", "datasets/coco/coco_val_2014_adaptive_withclasses-004.h5"]
	# n_captions_per_image = 17
	
	coco_dataset = COCO_Dataset(data_path,img_data_path,vocab_path,coco_class, mode="val",nseqs_per_video=n_captions_per_image, dataset_name="msvd", ngram=1, result_dir="temp")
	# sys.exit(0)
	print("The length of the dataset is: " + str(len(coco_dataset)))
	coco_dataloader = torch.utils.data.DataLoader(coco_dataset, batch_size=50, shuffle=False, num_workers=4, drop_last=True, collate_fn=my_collate)	
	def tensor2lang(d, i2w):
		B, T = d.shape
		return [" ".join([i2w[int(i.item())] for i in d[b] if i.item()>4]) for b in range(B)]#
	
	logfile = "coco_verb_rm.log"
	# logfile = "test.log"
	# if os.path.isfile(logfile):
	# 	os.remove(logfile)
	import logging
	logging.basicConfig(
		level=getattr(logging, "DEBUG"),
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			# logging.FileHandler("myresults/i3d_z_vatex_opsedo_2021_06_03_11_35_41/remove_verb.log"),
			logging.FileHandler(logfile),
			# logging.StreamHandler()
		]
	)
	logger = logging.getLogger(__name__)

	import pandas as pd
	from tqdm import tqdm
	df = pd.DataFrame()
	all_capts = []
	all_capts_m = []
	all_capts_o = []
	all_capts_id = []

	for i,data in tqdm(enumerate(coco_dataloader)):
		caps, caps_m, caps_o = [tensor2lang(torch.cat(data[i]),coco_dataset.get_i2w()) for i in [0,1,2]]
		seq_len_per_video = [len(i) for i in data[0]]
		caps_id = np.array([d for ix,d in enumerate(data[5]) for j in range(seq_len_per_video[ix])])
		for j in range(len(caps_id)):
			logger.debug('[%d] video caption: %s' %
					(caps_id[j], caps[j]))
			logger.debug('[%d] video context: %s' %
					(caps_id[j], caps_m[j]))
			logger.debug('[%d] video verb: %s' %
					(caps_id[j], caps_o[j]))
			all_capts.append(caps[j])
			all_capts_id.append(caps_id[j])

	df['video_id'] = all_capts_id
	df['video_caption'] = all_capts

	df.to_csv("datasets/msvd/annotations_msvd_val_rem_2.csv", index=False)