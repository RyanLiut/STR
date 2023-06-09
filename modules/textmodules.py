from __future__ import print_function, division
import torch
import torch.nn as nn
from build_vocab_coco import Vocabulary
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda:0")
import sys
sys.path.insert(0,'/disk2/liuzhu/DivCap/mycvae')
from modules.utils import masked_softmax, masked_mean, tiny_value_of_dtype

class PostFusion(nn.Module):
	"""Fusing posteriori z into inferred z"""
	def __init__(self, qk_dim, att_dim):
		super(PostFusion, self).__init__()
		self.qk_dim = qk_dim
		self.att_dim = att_dim
		self.fc_fusion = nn.Linear(qk_dim+att_dim, qk_dim) # The easiest way
		self.fc_k = nn.Linear(qk_dim, att_dim)
		self.fc_q = nn.Linear(qk_dim, att_dim)
		self.fc_att = nn.Linear(att_dim, 1)

		self.FC_fus = nn.Linear(qk_dim * 2, qk_dim)

	def forward(self, q, k=None, type='FC'):

		if type == "FC":
			return self.FC_fus(q), None
		else: # att
			x = self.fc_k(k)
			inputs = self.fc_q(q).unsqueeze(1) + x
			weights = F.softmax(self.fc_att(torch.tanh( inputs )), dim=1).permute(0,2,1)
			x_obj_agg = weights.bmm(x).squeeze(1) # another modification
			y = self.fc_fusion(torch.cat([x_obj_agg, q], dim=-1))
			return y, weights

class GaussPrior(nn.Module):
	"""docstring for GaussPrior"""
	def __init__(self, meanimfeats_size, vocab_size, sent_emd_size, sent_enc_size, max_length, z_indeps=False, seq_CVAE=False, scale_o=0.35):
		super(GaussPrior, self).__init__()

		self.z_indeps = z_indeps
		self.max_length = max_length
		self.emd = nn.Embedding(vocab_size, sent_emd_size)
		self.lstm_mask = nn.LSTMCell(meanimfeats_size + sent_enc_size + sent_emd_size, 1024, bias=True)#+ attention_dim
		if not self.z_indeps:
			self.lstm_obj = nn.LSTMCell(meanimfeats_size + 2*sent_enc_size + sent_emd_size, 256, bias=True)#+ attention_dim
		else:
			self.lstm_obj = nn.LSTMCell(meanimfeats_size + sent_enc_size + sent_emd_size, 256, bias=True)#+ attention_dim
		self.fc_m_out = nn.Linear(1024, 2*sent_enc_size)#.to(device)
		self.fc_o_out = nn.Linear(256, 2*sent_enc_size)#.to(device)
		self.sent_enc_size = sent_enc_size
		self.sent_emd_size = sent_emd_size
		self.seq_CVAE = seq_CVAE
		self.scale_o = scale_o

	def load_my_state_dict(self, state_dict):
		own_state = self.state_dict()
		for name, param in state_dict.items():
				if name not in own_state:
						 continue
				if isinstance(param, nn.Parameter):
						param = param.data
				own_state[name].copy_(param)

	def init_attn_lstm_hidden(self, bsz, hsize):
		h = Variable(torch.zeros(bsz, hsize), requires_grad=True).cuda()
		c = Variable(torch.zeros(bsz, hsize), requires_grad=True).cuda()
		return (h,c)	

	def get_kl_div_mask(self, obj_enc, x, q_means_mask, q_logs_mask, q_z_mask):	# q(z_t | z_t_1, x, I), p(z_t | z_t_1, x_t_1, I)
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)
		hiddens_mask = self.init_attn_lstm_hidden(q_means_mask.size(0), 1024)

		z_m_t_1 = torch.zeros(q_z_mask.size(0),q_z_mask.size(2)).cuda()
		seq_len = ((x != 0).float().cuda() * (x != 2).float().cuda())
		x = self.emd( x )
		#x_t_1 = torch.zeros(meanimfeats.size(0),self.sent_emd_size).cuda()
		p_means_mask = []
		p_logs_mask = []
		# here, logs means log of the variance

		#print('x ', x.size(), ' q_means ',q_means.size(),' q_logs ', q_logs.size(), ' q_z ', q_z.size())
		for i in range(q_z_mask.shape[1]):
			x_t_1 = x[:,i,:]
			hiddens_mask = self.lstm_mask(torch.cat([z_m_t_1,x_t_1,meanimfeats], dim=1), hiddens_mask)
			# Why the priori depends on the posteriori variable z_m_t_1 - Here z is a sequential hidden variable.
			# Why x_t_1, given x is the GT, while in prior the x cannot be GT?
			out = self.fc_m_out(hiddens_mask[0])
			p_means_mask.append( out[:,None,:self.sent_enc_size] )
			p_logs_mask.append( out[:,None,self.sent_enc_size:] )
			z_m_t = q_z_mask[:,i,:] 

			z_m_t_1 = q_z_mask[:,i,:] 


		p_means_mask = torch.cat(p_means_mask, dim=1)
		p_logs_mask = torch.cat(p_logs_mask, dim=1)

		kl_div_mask = 0.5*(p_logs_mask - q_logs_mask) + (q_logs_mask.exp() +(q_means_mask - p_means_mask)**2 )/(2*p_logs_mask.exp()) - 0.5
		kl_div_mask = torch.sum(kl_div_mask, dim=2)
		kl_div_mask = kl_div_mask * seq_len
		kl_div_mask = torch.sum(kl_div_mask, dim=1)

		return kl_div_mask

	def get_kl_div(self, obj_enc, x, q_means_mask, q_logs_mask, q_z_mask, q_means_obj, q_logs_obj, q_z_obj):	# q(z_t | z_t_1, x, I), p(z_t | z_t_1, x_t_1, I)
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)
		hiddens_mask = self.init_attn_lstm_hidden(q_means_mask.size(0), 1024)
		hiddens_obj = self.init_attn_lstm_hidden(q_means_obj.size(0), 256)

		z_m_t_1 = torch.zeros(q_z_mask.size(0),q_z_mask.size(2)).cuda()
		z_o_t_1 = torch.zeros(q_z_obj.size(0),q_z_obj.size(2)).cuda()
		seq_len = ((x != 0).float().cuda() * (x != 2).float().cuda())

		x = self.emd( x )
		p_means_mask = []
		p_logs_mask = []
		p_means_obj = []
		p_logs_obj = []
		for i in range(q_z_mask.shape[1]):
			x_t_1 = x[:,i,:]
			hiddens_mask = self.lstm_mask(torch.cat([z_m_t_1,x_t_1,meanimfeats], dim=1), hiddens_mask)
			out = self.fc_m_out(hiddens_mask[0])
			p_means_mask.append( out[:,None,:self.sent_enc_size] )
			p_logs_mask.append( out[:,None,self.sent_enc_size:] )
			z_m_t = q_z_mask[:,i,:] 

			if not self.z_indeps:
				hiddens_obj = self.lstm_obj(torch.cat([z_m_t,z_o_t_1,x_t_1,meanimfeats], dim=1), hiddens_obj)
			else:
				hiddens_obj = self.lstm_obj(torch.cat([z_o_t_1,x_t_1,meanimfeats], dim=1), hiddens_obj)
			out = self.fc_o_out(hiddens_obj[0])
			p_means_obj.append( out[:,None,:self.sent_enc_size] )
			p_logs_obj.append( out[:,None,self.sent_enc_size:] )

			z_m_t_1 = q_z_mask[:,i,:] 
			z_o_t_1 = q_z_obj[:,i,:]


		p_means_mask = torch.cat(p_means_mask, dim=1)
		p_logs_mask = torch.cat(p_logs_mask, dim=1)

		p_means_obj = torch.cat(p_means_obj, dim=1)
		p_logs_obj = torch.cat(p_logs_obj, dim=1)

		# ADD: sample z from prior to calculate the validation loss
		p_z_m = (torch.randn(p_means_mask.size()).cuda())*torch.exp(0.35*p_logs_mask) + p_means_mask
		p_z_o = (torch.randn(p_means_obj.size()).cuda())*torch.exp(0.35*p_logs_obj) + p_means_obj
		p_z = torch.cat([p_z_m,p_z_o], dim=2)

		kl_div_mask = 0.5*(p_logs_mask - q_logs_mask) + (q_logs_mask.exp() +(q_means_mask - p_means_mask)**2 )/(2*p_logs_mask.exp()) - 0.5
		kl_div_mask = torch.sum(kl_div_mask, dim=2)
		kl_div_mask = kl_div_mask * seq_len
		kl_div_mask = torch.sum(kl_div_mask, dim=1)

		kl_div_obj = 0.5*(p_logs_obj - q_logs_obj) + (q_logs_obj.exp() +(q_means_obj - p_means_obj)**2 )/(2*p_logs_obj.exp()) - 0.5
		kl_div_obj = torch.sum(kl_div_obj, dim=2)
		kl_div_obj = kl_div_obj * seq_len
		kl_div_obj = torch.sum(kl_div_obj, dim=1)
		return (kl_div_mask + kl_div_obj), {"p_means_mask":round(p_means_mask.mean().item(),3), "p_logs_mask":round(p_logs_mask.mean().item(),3), "p_means_obj":round(p_means_obj.mean().item(),3), "p_logs_obj":round(p_logs_obj.mean().item(),3)} 

	def get_sample_t(self, obj_enc, x_t_1, z_t_1, hiddens):
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)



		if hiddens is None:
			hiddens_mask = self.init_attn_lstm_hidden(meanimfeats.size(0),1024)
			hiddens_obj = self.init_attn_lstm_hidden(meanimfeats.size(0),256)
		else:
			hiddens_mask = hiddens[0]
			hiddens_obj = hiddens[1]

		if x_t_1 is None:
			x_t_1 =torch.ones(meanimfeats.size(0),1).long().cuda()	

		if z_t_1 is None:
			z_t_1_m = torch.zeros(meanimfeats.size(0),self.sent_enc_size).cuda()
			z_t_1_o = torch.zeros(meanimfeats.size(0),self.sent_enc_size).cuda()
		else:
			if self.seq_CVAE:
				z_t_1_m = z_t_1
				z_t_1_o = z_t_1
			else:
				z_t_1_m = z_t_1[:,:z_t_1.size(1)//2]
				z_t_1_o = z_t_1[:,z_t_1.size(1)//2:]
		x_t_1 = self.emd(x_t_1)[:,0,:]
		hiddens_mask = self.lstm_mask(torch.cat([z_t_1_m,x_t_1,meanimfeats], dim=1), hiddens_mask)
		out = self.fc_m_out(hiddens_mask[0])
		p_means_m = out[:,:self.sent_enc_size]
		p_logs_m = out[:,self.sent_enc_size:]
		ff = 1
		p_logs_m = torch.ones_like(p_logs_m) *ff
		# DEBUG: Fix the z_m
		p_z_m = (torch.randn(p_means_m.size()).cuda())*torch.exp(0.35*p_logs_m) + p_means_m
		# p_z_m = p_means_m

		if not self.z_indeps:
			hiddens_obj = self.lstm_obj(torch.cat([p_z_m,z_t_1_o,x_t_1,meanimfeats], dim=1), hiddens_obj)
		else:
			hiddens_obj = self.lstm_obj(torch.cat([z_t_1_o,x_t_1,meanimfeats], dim=1), hiddens_obj)
		out = self.fc_o_out(hiddens_obj[0])
		p_means_o = out[:,:self.sent_enc_size]
		p_logs_o = out[:,self.sent_enc_size:]
		# DEBUG: Fix the z_o
		# change the variance
		p_logs_o = torch.ones_like(p_logs_o) * ff
		# DEBUG 02: change the p_z_o_mean
		# p_means_o = torch.normal(0,1,size=p_means_o.size()).cuda()
		# p_logs_o = torch.normal(0,1,size=p_logs_o.size()).cuda()
		p_z_o = (torch.randn(p_means_o.size()).cuda())*torch.exp(self.scale_o*p_logs_o) + p_means_o
		# p_z_o = p_means_o

		if self.seq_CVAE:
			return p_z_m, (hiddens_mask,hiddens_obj)
		else:
			return torch.cat([p_z_m,p_z_o], dim=1), (hiddens_mask,hiddens_obj) 

	def get_sample_obj_t(self, obj_enc, x_t_1, z_t_m, z_t_1_o, hiddens_obj):
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)

		if hiddens_obj is None:
			hiddens_obj = self.init_attn_lstm_hidden(meanimfeats.size(0),256)

		if x_t_1 is None:
			x_t_1 = torch.ones(meanimfeats.size(0),1).long().cuda()	

		if z_t_1_o is None:
			z_t_1_o = torch.zeros(meanimfeats.size(0),self.sent_enc_size).cuda()
		
		x_t_1 = self.emd(x_t_1)[:,0,:]

		if not self.z_indeps:
			hiddens_obj = self.lstm_obj(torch.cat([z_t_m,z_t_1_o,x_t_1,meanimfeats], dim=1), hiddens_obj)
		else:
			hiddens_obj = self.lstm_obj(torch.cat([z_t_1_o,x_t_1,meanimfeats], dim=1), hiddens_obj)
		out = self.fc_o_out(hiddens_obj[0])
		p_means = out[:,:self.sent_enc_size]
		p_logs = out[:,self.sent_enc_size:]
		p_z_o = (torch.randn(p_means.size()).cuda())*torch.exp(0.3*p_logs) + p_means
		

		
		return p_z_o, hiddens_obj


	def forward(self, obj_enc, x, p_z_t_1, hiddens, q_means_mask, q_logs_mask, q_z_mask, q_means_obj, q_logs_obj, q_z_obj, train=False, reverse=False):
		if not reverse and train:
			if q_means_obj is not None and q_logs_obj is not None and q_z_obj is not None:
				return self.get_kl_div(obj_enc, x, q_means_mask, q_logs_mask, q_z_mask, q_means_obj, q_logs_obj, q_z_obj) 
			else:
				return self.get_kl_div_mask(obj_enc, x, q_means_mask, q_logs_mask, q_z_mask) 
		elif reverse and train:
			return self.get_sample_obj_t(obj_enc, x, q_z_mask, p_z_t_1, hiddens) 
		else:
			return self.get_sample_t(obj_enc, x, p_z_t_1, hiddens)


class TextEncoderMask(nn.Module):
	def __init__(self, word_dim, embed_size, maxlength, meanimfeats_size, vocab_size, mask_hidden_size= 1024,  num_layers=1, use_abs=False, bidirectional = True, dropout = 0.0, use_pos=False):
		super().__init__()
		self.emd = nn.Embedding(vocab_size, word_dim)
		# In context encoder, there is no image features to use.
		self.gru = nn.GRU(word_dim, mask_hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)#.to(device);
		self.use_pos = use_pos
		if use_pos:
			self.fc1_out = nn.Linear(embed_size + 3*mask_hidden_size, 2*embed_size)
		else:
			self.fc1_out = nn.Linear(embed_size + 2*mask_hidden_size, 2*embed_size)#.to(device)
		self.embed_size = embed_size
		self.mask_hidden_size = mask_hidden_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.relu = nn.ReLU()

		self.max_length = maxlength

	def init_hidden(self, bsz,hidsize):
		h = Variable(torch.zeros(self.num_layers * (2 if self.bidirectional else 1), bsz, hidsize), requires_grad=True).cuda()
		return h

	def forward_BiGRU(self, x, lengths, posHidden=None):

		x = self.emd(x)
		hiddens = self.init_hidden(x.data.size(0),self.mask_hidden_size)
		emb = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
		#self.gru.flatten_parameters()
		outputs, hidden_t = self.gru(emb, hiddens)
		output_unpack = pad_packed_sequence(outputs, batch_first=True)
		#print('output_unpack[0] ',output_unpack[0].size())
		#What does hidden_o mean? And shape?
		hidden_o = output_unpack[0]#.sum(1)
		#print('hidden_o ',hidden_o.size())
		## What do "hidden_o, b, f" mean?
		hidden_f = hidden_o[:,:,:self.mask_hidden_size]
		hidden_b = hidden_o[:,:,self.mask_hidden_size:]

		hidden_o = hidden_o.sum(1)
		output_lengths = Variable(lengths.float().cuda(), requires_grad=False)
		hidden_o = torch.div(hidden_o, output_lengths.view(-1, 1))
		hidden_o = (hidden_o[:,:self.mask_hidden_size] + hidden_o[:,self.mask_hidden_size:])/2

		
		means = torch.zeros(x.data.size(0), self.max_length, self.embed_size).cuda()
		logs = torch.zeros(x.data.size(0), self.max_length, self.embed_size).cuda()
		z = torch.zeros(x.data.size(0), self.max_length, self.embed_size).cuda()
		z_t_1 = torch.zeros(x.size(0), self.embed_size).cuda()

		for t in range(max(lengths)):
			
			if not posHidden is None and not self.use_pos is None:
				output = self.fc1_out(torch.cat([hidden_f[:,t,:], hidden_b[:,t,:], posHidden, z_t_1],dim=1))
			else:
				output = self.fc1_out(torch.cat([hidden_f[:,t,:], hidden_b[:,t,:],z_t_1],dim=1))

			mean = output[:,:output.size(1)//2]
			log = output[:,output.size(1)//2:]
			std = torch.exp(0.5 * log)

			z_t = mean + (torch.randn(mean.size()).cuda())*std # "reparameterization trick!"

			means[:,t,:] = mean
			logs[:,t,:] = log
			z[:,t,:] = z_t

			z_t_1 = z_t#.clone()

		return means, logs, z, hidden_o

	def forward(self, x, lengths, posHidden=None):
		out = self.forward_BiGRU(x, lengths, posHidden)
		return out	

	def load_my_state_dict(self, state_dict):
		own_state = self.state_dict()
		for name, param in state_dict.items():
				if name not in own_state:
						 continue
				if isinstance(param, nn.Parameter):
						param = param.data
				own_state[name].copy_(param)		


class TextEncoderObj(nn.Module):
	def __init__(self, word_dim, embed_size, maxlength, meanimfeats_size,vocab_size, mask_hidden_size, num_layers=1, dropout = 0.0, z_indeps=False, scale_o=0.5):
		super().__init__()
		bidirectional = False
		self.emd = nn.Embedding(vocab_size, word_dim)
		self.z_indeps = z_indeps
		self.hidden_dim = 256
		self.att_dim = 64
		self.att_dim = self.hidden_dim
		self.fc2_q = nn.Linear(embed_size, self.att_dim)
		if not self.z_indeps:
			self.fc2_k = nn.Linear(word_dim + meanimfeats_size + mask_hidden_size, self.att_dim)
			# self.gru = nn.GRU(word_dim + meanimfeats_size + mask_hidden_size, self.hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)#.to(device);
			self.fc1_out = nn.Linear(self.hidden_dim + 2*embed_size, 2*embed_size) # not conditioned that tightly
			self.fc2_att = nn.Linear(self.att_dim, 1) # additive attention
		else:
			self.fc2_k = nn.Linear(word_dim + meanimfeats_size, self.hidden_dim)
			# self.gru = nn.GRU(word_dim + meanimfeats_size, self.hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
			self.fc1_out = nn.Linear(self.hidden_dim + embed_size, 2*embed_size)
		#self.fc2_out = nn.Linear(256, 2*embed_size)#.to(device)
		self.fc3 = nn.Linear(word_dim + meanimfeats_size + mask_hidden_size, self.hidden_dim)
		self.embed_size = embed_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.relu = nn.ReLU()

		self.max_length = maxlength
		self.scale_o = scale_o

	def init_hidden(self, bsz,hidsize):
		h = Variable(torch.zeros(self.num_layers * (2 if self.bidirectional else 1), bsz, hidsize), requires_grad=True).cuda()
		return h

	def load_my_state_dict(self, state_dict):
		own_state = self.state_dict()
		for name, param in state_dict.items():
				if name not in own_state:
						 continue
				if isinstance(param, nn.Parameter):
						param = param.data
				own_state[name].copy_(param)

	def forward_BiGRU(self, x_obj, obj_enc, mask_hidden, mask_z,  lengths):

		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)

		hiddens = self.init_hidden(x_obj.data.size(0),self.hidden_dim)

		x_obj = self.emd(x_obj)
		meanimfeats = meanimfeats.unsqueeze(1).repeat(1,20,1) # 20 objects
		mask_hidden = mask_hidden.unsqueeze(1).repeat(1,20,1)
		#print('x_obj ',x_obj.size(), ' meanimfeats ', meanimfeats.size(), ' mask_hidden ', mask_hidden.size())
		if not self.z_indeps:
			x_obj = torch.cat([x_obj,meanimfeats,mask_hidden],dim=2)
		else:
			x_obj = torch.cat([x_obj, meanimfeats], dim=2)

		'''
		Original code version: temporal dependency for objects
		Change to: additive attention
		x_obj = pack_padded_sequence(x_obj, lengths.cpu(), batch_first=True)
		outputs, hidden_t = self.gru(x_obj, hiddens)
		output_unpack = pad_packed_sequence(outputs, batch_first=True)

		hidden_o = output_unpack[0].sum(1) # Here it could use attention, not just sum. And lstm is not nessecary.
		output_lengths = Variable(lengths.float().cuda(), requires_grad=False)
		hidden_o = torch.div(hidden_o, output_lengths.view(-1, 1))
		'''

		z_o_t_1 = torch.zeros(mask_z.size(0),self.embed_size).cuda()
		means =  torch.zeros(mask_z.size(0),self.max_length,self.embed_size).cuda()
		logs =  torch.zeros(mask_z.size(0),self.max_length,self.embed_size).cuda()
		obj_z =  torch.zeros(mask_z.size(0),self.max_length,self.embed_size).cuda()

		weights_all = []

		for t in range(self.max_length):
			
			if not self.z_indeps:
				# From B*N*d to B*d and remove the temporal dependency.
				# [WARNING] I did not remove the side-effect of mask
				x_obj_2 = self.fc2_k(x_obj)
				inputs = self.fc2_q(mask_z[:,t,:]).unsqueeze(1) + x_obj_2
				weights = F.softmax(self.fc2_att(torch.tanh( inputs )), dim=1).permute(0,2,1)
				x_obj_agg = weights.bmm(x_obj).squeeze(1) # another modification
				weights_all.append(weights.squeeze(1))

				x_obj_agg = self.fc3(x_obj_agg)

				output = self.fc1_out(torch.cat([x_obj_agg, mask_z[:,t,:],z_o_t_1],dim=1))
				# output = self.fc1_out(torch.cat([hidden_o, mask_z[:,t,:],z_o_t_1],dim=1))
			else:
				x_obj_agg = self.fc2_k(x_obj).mean(1)
				output = self.fc1_out(torch.cat([x_obj_agg, z_o_t_1],dim=1))

			mean = output[:,:output.size(1)//2]
			log = output[:,output.size(1)//2:]
			std = torch.exp(0.5 * log) # Test the effecs of std.
			# std = torch.exp(self.scale_o * log)

			z_o_t = mean + (torch.randn(mean.size()).cuda())*std

			means[:,t,:] = mean
			logs[:,t,:] = log
			obj_z[:,t,:] = z_o_t

			z_o_t_1 = z_o_t#.clone()
		return means, logs, obj_z, weights_all

	def forward(self, x_obj, obj_enc, mask_hidden, mask_z,  lengths):
		out = self.forward_BiGRU(x_obj, obj_enc, mask_hidden, mask_z, lengths)
		return out


class AttentionLSTM(nn.Module):
	"""
	Attention Network.
	"""
	# New input: (batch_size, pixels, concepts) # pixels == max_length, concepts == embed_size
	# Thus: (batch_size, max_length, embed_size)

	def __init__(self, meanimfeats_size, sent_enc_size, seq_size, lang_lstm_size, obj_enc_size, attention_dim, use_z=True):#embed_size, hidden_size, 
		"""
		:param encoder_dim: feature size of encoded images
		:param decoder_dim: size of decoder's RNN
		:param attention_dim: size of the attention network
		"""
		super().__init__()
		
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
		self.attention_dim = attention_dim
		self.use_z = use_z


		#self.meanimfeats_embed = nn.Linear(meanimfeats_size, hidden_size)
		#self.seq_embed = nn.Linear(seq_size,hidden_size)
		#self.lang_lstm_embed = nn.Linear(lang_lstm_size,hidden_size)


		## DEBUG 01: what if no latent variable z?
		if self.use_z:
			self.attn_lstm = nn.LSTMCell(meanimfeats_size + seq_size + lang_lstm_size + sent_enc_size, attention_dim, bias=True)
		else:
			self.attn_lstm = nn.LSTMCell(meanimfeats_size + seq_size + lang_lstm_size, attention_dim, bias=True)
		self.obj_enc_attn = nn.Linear(obj_enc_size, attention_dim)

		self.attn_out_fc = nn.Linear(attention_dim,attention_dim)

		self.final_attn_fc = nn.Linear( attention_dim, 1)

	def init_attn_lstm_hidden(self, bsz):
		h = Variable(torch.zeros(bsz, self.attention_dim), requires_grad=True).cuda()
		c = Variable(torch.zeros(bsz, self.attention_dim), requires_grad=True).cuda()
		return (h,c)	

	def forward(self, meanimfeats, sent_enc, obj_enc, obj_enc_mask, seq_t, lang_lstm_hidden_t_1, attn_lstm_hidden = None):
		"""
		Forward propagation.
		:param encoder_out: encoded images, a tensor of dimension (batch_size, max_length, embed_size)
		:param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, hidden_size)
		:return: attention weighted encoding, weights
		"""
		if attn_lstm_hidden is None:
			attn_lstm_hidden = self.init_attn_lstm_hidden(meanimfeats.size(0))
		else:
			_h = attn_lstm_hidden[0]
			_c = attn_lstm_hidden[1]
			attn_lstm_hidden = (_h[:meanimfeats.size(0)],_c[:meanimfeats.size(0)])

		#print('meanimfeats ', meanimfeats.size(), ' seq_t ',seq_t.size(),' lang_lstm_hidden_t_1 ',lang_lstm_hidden_t_1.size())
		#_meanimfeats = self.meanimfeats_embed(meanimfeats)
		#seq_t = self.seq_embed(seq_t)
		## DEBUG 01: what if no latent variable z?
		# Expected: captions for one video should be the same!
		if self.use_z:
			attn_in = torch.cat([meanimfeats,sent_enc,seq_t,lang_lstm_hidden_t_1],dim=1)#,attn_lstm_hidden[0]
		else:
			attn_in = torch.cat([meanimfeats,seq_t,lang_lstm_hidden_t_1],dim=1)

		attn_lstm_hidden = self.attn_lstm(attn_in, attn_lstm_hidden)

		obj_enc_attn_out = self.obj_enc_attn(obj_enc)

		lstm_attn_out = self.attn_out_fc(attn_lstm_hidden[0])

		attn_out = self.final_attn_fc( self.tanh(obj_enc_attn_out + lstm_attn_out.unsqueeze(1).repeat(1,obj_enc_attn_out.size(1),1) ))
		attn_out = attn_out.squeeze(-1)
		#alpha = self.softmax(attn_out)
		alpha = masked_softmax(attn_out,obj_enc_mask)
		'''att1 = self.encoder_att(encoder_out)  # (batch_size, max_length, attention_dim)
		att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
		att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, max_length)
		alpha = self.softmax(att)  # (batch_size, max_length)
		attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, embed_size)'''

		return attn_lstm_hidden, alpha	
		    				    

class TextDecoder(nn.Module):
	"""
	Decoder.
	"""
	def __init__(self, sent_enc_size, vocab_size, decoder_dim=1024, decoder_embed_dim=1000, encoder_dim=2048, attention_dim=2048, SC_dim=None, SC_n_cls=None, dropout=0.2, use_z=True, debug=False, logger=None):
		"""
		:param attention_dim: size of attention network, also the hidden size of the first layer(attention layer) lstm
		:param embed_dim/decoder_embed_dim: embedding size
		:param decoder_dim: size of decoder's RNN (the second layer)
		:param vocab_size: size of vocabulary
		:param encoder_dim: feature size of encoded images (objects)
		:param dropout: dropout
		"""
		super().__init__()
		self.encoder_dim = encoder_dim
		self.attention_dim = attention_dim
		self.decoder_embed_dim = decoder_embed_dim
		self.decoder_dim = decoder_dim
		self.vocab_size = vocab_size
		self.dropout = dropout
		self.bidirectional = False
		self.word_dropout_rate = 0.50
		self.embedding_dropout = nn.Dropout(p=0.0)
		self.latent_dim = sent_enc_size
		self.SC_dim = SC_dim # use semantic concepts
		self.attention = AttentionLSTM(meanimfeats_size=encoder_dim, sent_enc_size=sent_enc_size, seq_size=decoder_embed_dim, lang_lstm_size=decoder_dim, obj_enc_size=encoder_dim, attention_dim=attention_dim, use_z=use_z) # this is not good
		if self.SC_dim: # get logits
			self.SC_proj = nn.Sequential(
			nn.Linear(encoder_dim, self.SC_dim), nn.ReLU(), nn.Linear(self.SC_dim, SC_n_cls)
		)

		self.embedding = nn.Embedding(vocab_size, decoder_embed_dim).to(device)  # embedding layer
		self.dropout = nn.Dropout(p=self.dropout)
		self.decode_step = nn.LSTMCell( encoder_dim + attention_dim + decoder_dim, decoder_dim, bias=True)  # decoding LSTMCell 
		self.output_projection = nn.Sequential(
				nn.Linear(decoder_dim, decoder_embed_dim), nn.Tanh()
			)
		self.output_layer = nn.Linear(decoder_embed_dim, vocab_size, bias=False)
		self.debug = debug
		self.logger = logger

		if self.debug:
			self.logger.warning("The validation is DEBUGGING!")

	def load_my_state_dict(self, state_dict):
		own_state = self.state_dict()
		for name, param in state_dict.items():
				if name not in own_state:
						 continue
				if isinstance(param, nn.Parameter):
						param = param.data
				own_state[name].copy_(param)
		
	
	def init_hidden_state(self,):
		h = Variable(torch.zeros(self.batch_size, self.decoder_dim), requires_grad=True).cuda()
		c = Variable(torch.zeros(self.batch_size, self.decoder_dim), requires_grad=True).cuda()
		return (h,c)
	

	def _train(self, seq, meanimfeats, sent_enc, obj_enc, caption_lengths, prior_model):
		"""
		Forward propagation.
		:param sent_enc: the latent variable z (z_m + z_o), a tensor of dimension (batch_size, seq_len, z_dim)
		:param obj_enc: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
		:(enc_image_size, enc_image_size)? Why? Should not be (obj_num)
		:param seq: encoded captions, a tensor of dimension (batch_size, max_caption_length)
		:param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
		:return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
		"""

		self.batch_size = obj_enc.size(0)
		encoder_dim = obj_enc.size(1)
		vocab_size = self.vocab_size

		# Flatten image
		# (batch_size, num_pixels, encoder_dim) num_pixels == max_length
		#print('obj_enc ', obj_enc.size())
		max_length = seq.size(1)
		org_seq = seq.clone()

		# Sort input data by decreasing lengths; why? apparent below
		'''caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
		encoder_out = encoder_out[sort_ind]
		encoded_captions = encoded_captions[sort_ind]'''

		# Embedding
		#embeddings = self.embedding(seq)  # (batch_size, max_caption_length, decoder_embed_dim)
		if self.word_dropout_rate > 0:
			prob = torch.rand(seq.size()).cuda()
			prob[(seq.data - 1) * (seq.data - 0) * (seq.data - 2) == 0] = 1
			decoder_input_sequence = seq.clone()
			decoder_input_sequence[prob < self.word_dropout_rate] = 0#3
			#input_embedding = self.emd(decoder_input_sequence)
			embeddings = self.embedding(decoder_input_sequence)
			embeddings = self.embedding_dropout(embeddings)
		else:
			embeddings = self.embedding(seq)

		# Initialize LSTM state
		h, c = self.init_hidden_state()#obj_enc  # (batch_size, decoder_dim)

		# We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
		# So, decoding lengths are actual lengths - 1
		decode_lengths = caption_lengths.tolist()#(caption_lengths - 0).tolist() #.to(device)

		# Create tensors to hold word predicion scores and alphas
		predictions = torch.zeros(self.batch_size, max_length-1, vocab_size).cuda()#max(decode_lengths)
		alphas = torch.zeros(self.batch_size, max_length-1, obj_enc.size(1)).cuda()#max(decode_lengths)

		# At each time-step, decode by
		# attention-weighing the encoder's output based on the decoder's previous hidden state output
		# then generate a new word in the decoder with the previous word and the attention weighted encoding
		#meanimfeats, seq_t, lang_lstm_hidden_t_1, 
		attn_lstm_hidden = None
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		#print('obj_enc_mask ',obj_enc_mask[0])
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)
		#spt_feat = self.get_spt_feat(meanimfeats, obj_enc)
		hiddens_obj = None
		sent_enc_o_t = None
		all_weighted_encodings = []
		for t in range(max(decode_lengths)):
			batch_size_t = sum([l > t for l in decode_lengths])
			#print(' embeddings ', embeddings.size())

			if prior_model is None:
				attn_lstm_hidden, alpha = self.attention( meanimfeats[:batch_size_t], sent_enc[:batch_size_t, t,:], obj_enc[:batch_size_t], 
					obj_enc_mask[:batch_size_t], embeddings[:batch_size_t, t, :], h[:batch_size_t], attn_lstm_hidden)
				# here the sent_enc shares the same t with decoder to show the nature of recontruction.
			else:
				# The prior model is used in train stage when object information is unavailable in the pseudo supervision.
				x_t_1 = org_seq[:batch_size_t,t-1:t] if t > 0 else None
				sent_enc_o_t, hiddens_obj = prior_model(obj_enc=obj_enc[:batch_size_t, :], x=x_t_1, 
					p_z_t_1=sent_enc_o_t if sent_enc_o_t is None else sent_enc_o_t[:batch_size_t,:], 
					hiddens=hiddens_obj if hiddens_obj is None else (hiddens_obj[0][:batch_size_t],hiddens_obj[1][:batch_size_t]), 
					q_means_mask=None, q_logs_mask=None, q_z_mask=sent_enc[:batch_size_t, t,:], q_means_obj=None, q_logs_obj=None, q_z_obj=None,
					train=True, reverse=True)
			
				q_z_hybrid = torch.cat([sent_enc[:batch_size_t, t,:], sent_enc_o_t[:batch_size_t,:]], dim=1)

				# How to deal with the dim of q_z_hybrid is not consistent with that of q_z, while they share the same self.attention?
				attn_lstm_hidden, alpha = self.attention( meanimfeats[:batch_size_t], q_z_hybrid, obj_enc[:batch_size_t], 
					obj_enc_mask[:batch_size_t], embeddings[:batch_size_t, t, :], h[:batch_size_t], attn_lstm_hidden)

			attention_weighted_encoding = (obj_enc[:batch_size_t] * alpha.unsqueeze(2)).sum(dim=1)

			# Why here concat h into the first part of LSTM?
			h, c = self.decode_step(
				torch.cat([attention_weighted_encoding, attn_lstm_hidden[0], h[:batch_size_t]], dim=1),
				(h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
			#preds = self.fc(h)  #self.dropout(h), (batch_size_t, vocab_size) 

			_h = self.output_projection(h)
			preds = self.output_layer(_h)
			predictions[:batch_size_t, t, :] = preds#nn.functional.log_softmax(preds)
			alphas[:batch_size_t, t, :] = alpha # alphas can be used to visualization.
		ent = None	
		if self.SC_dim:
			SC_pred = self.SC_proj(torch.bmm(alphas, obj_enc).mean(dim=1))
			return predictions, alphas, ent, SC_pred
		else:
			return predictions, alphas, ent  


	def beam_search(self, obj_enc, prior_model, max_length=20, beam_width=2, group_size=1, diversity_lambda=0.8, post_z=None):
		max_num = max(5, beam_width)
		self.batch_size = obj_enc.size(0)
		# index 1 is <SOS>
		words = Variable(torch.Tensor([1]).long(), requires_grad=False).repeat(self.batch_size).view(self.batch_size, 1, 1)
		probs = Variable(torch.zeros(self.batch_size, 1))  # [batch, beam]
		if torch.cuda.is_available():
			words = words.cuda()
			probs = probs.cuda()

		h, c = self.init_hidden_state()
		h = h.unsqueeze(0)
		c = c.unsqueeze(0)

		h_attn, c_attn = self.attention.init_attn_lstm_hidden(obj_enc.size(0))
		h_attn = h_attn.unsqueeze(0)
		c_attn = c_attn.unsqueeze(0)

		h_priorm, c_priorm = prior_model.init_attn_lstm_hidden(obj_enc.size(0), 1024)
		h_priorm = h_priorm.unsqueeze(0)
		c_priorm = c_priorm.unsqueeze(0)

		h_prioro, c_prioro = prior_model.init_attn_lstm_hidden(obj_enc.size(0), 256)
		h_prioro = h_prioro.unsqueeze(0)
		c_prioro = c_prioro.unsqueeze(0)

		z_t_1 = torch.zeros(obj_enc.size(0),self.latent_dim).cuda()
		z_t_1 = z_t_1.unsqueeze(0)
		
		all_hidden = [h.unsqueeze(3) for _ in range(group_size)]  # [1, batch, lstm_dim, beam]
		all_cell = [c.unsqueeze(3) for _ in range(group_size)]
		all_h_attn = [h_attn.unsqueeze(3) for _ in range(group_size)]
		all_c_attn = [c_attn.unsqueeze(3) for _ in range(group_size)]
		all_h_priorm = h_priorm.unsqueeze(3)
		all_c_priorm = c_priorm.unsqueeze(3) 
		all_h_prioro = h_prioro.unsqueeze(3)
		all_c_prioro = c_prioro.unsqueeze(3)
		all_z_t_1 = z_t_1.unsqueeze(3)

		all_z_list = []
		all_h_priorm_list = []
		all_h_prioro_list = []

		all_words = [words for _ in range(group_size)]  # [batch, length, beam] # how to extend in the length dim?
		all_probs = [probs for _ in range(group_size)] # [batch, beam]
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)

		done_beams = torch.zeros(self.batch_size, max_length, beam_width*group_size, dtype=int)
		done_beams_sum_prob = torch.zeros(self.batch_size, beam_width*group_size)
		i_e = [0 for _ in range(self.batch_size)]

		for t in range(max_length-1):

			for g in range(group_size):
				# Diverse Beam Search, 
				# if group_size == 1, it is equal to BS.
				new_words = []
				new_hidden = []
				new_cell = []
				new_hidden_attn = []
				new_cell_attn = []

				new_hidden_priorm = []
				new_cell_priorm = []
				new_hidden_prioro = []
				new_cell_prioro = []
				new_z_t_1_prior = []


				new_probs = []
				# self.logger.debug('t ',t,' all_words ',all_words.size())
				tmp_words = all_words[g].split(1, 2)
				# self.logger.debug('t ',t,' tmp_words ',len(tmp_words), ' lens ',[c.size() for c in tmp_words])
				tmp_probs = all_probs[g].split(1, 1)
				# self.logger.debug('t ',t,' all_hidden ',all_hidden.size())
				tmp_hidden = all_hidden[g].split(1, 3)
				tmp_cell = all_cell[g].split(1, 3)

				tmp_h_attn = all_h_attn[g].split(1, 3)
				tmp_c_attn = all_c_attn[g].split(1, 3)

				tmp_h_priorm = all_h_priorm.split(1, 3)
				tmp_c_priorm = all_c_priorm.split(1, 3)
				tmp_h_prioro = all_h_prioro.split(1, 3)
				tmp_c_prioro = all_c_prioro.split(1, 3)
				tmp_z_t_1 = all_z_t_1.split(1, 3)

				for i in range(len(tmp_words)):
					# self.logger.debug('This is the time %d', t) 
					# self.logger.debug('This is the tmp word %d', i)
					#print('i_t ',i,' tmp_hidden ',tmp_hidden[0].size())
					last_word = tmp_words[i].split(1, 1)[-1].view(self.batch_size,1)

					last_state_prior = ((tmp_h_priorm[i][0].squeeze(2),tmp_c_priorm[i][0].squeeze(2)),
						(tmp_h_prioro[i][0].squeeze(2),tmp_c_prioro[i][0].squeeze(2)))
					last_z_t_1 = tmp_z_t_1[i][0].squeeze(2)

					z_t, prior_hidden = prior_model(obj_enc = obj_enc, x=last_word, p_z_t_1=tmp_z_t_1[i][0].squeeze(2), 
						hiddens=last_state_prior, q_means_mask=None, q_logs_mask=None, q_z_mask=None, 
						q_means_obj=None, q_logs_obj=None, q_z_obj=None,train=False,reverse=True)

					# DEBUG: print out the z_t in different sample process, check out how it varies
					# New Func: to get the posteriori variable, if any. (for the sake of train debugging)
					if not post_z is None:
						z_t = torch.Tensor(post_z[:, t, :]).cuda()

					((hidden_priorm, states_priorm),(hidden_prioro, states_prioro)) = prior_hidden
					hidden_priorm = hidden_priorm.unsqueeze(0)
					states_priorm = states_priorm.unsqueeze(0)
					hidden_prioro = hidden_prioro.unsqueeze(0)
					states_prioro = states_prioro.unsqueeze(0)
					z_t = z_t.unsqueeze(0)
					# if self.debug:
					# r1, r2 = 0, 10
					# z_t = torch.FloatTensor(z_t.shape).normal_(r1, r2).cuda()
					# 	assert z_t.shape[1] == 1, "in the debug stage, the batch size should be one." # BS == 1
					# 	# self.logger.debug("Time step: {}".format(t))
					# 	# self.logger.debug("Tmp words: {}".format(tmp_words))
					# 	# self.logger.debug("Sampled z_t: {}".format(z_t[0,0,:]))
						# self.logger.debug("The last word is %d", last_word[0].item())
						# self.logger.debug(torch.mean(z_t[0]))


					embeddings = self.embedding(last_word)

					last_state_attn = (tmp_h_attn[i][0].squeeze(2),tmp_c_attn[i][0].squeeze(2))
					# The first-layer decoder.
					attn_lstm_hidden, alpha = self.attention( meanimfeats, z_t.squeeze(0), obj_enc, obj_enc_mask, 
						embeddings[:, 0, :], tmp_hidden[i][0].squeeze(2), last_state_attn)

					hidden_attn, states_attn = attn_lstm_hidden
					hidden_attn = hidden_attn.unsqueeze(0)
					states_attn = states_attn.unsqueeze(0)

					attention_weighted_encoding = (obj_enc * alpha.unsqueeze(2)).sum(dim=1)

					last_state = (tmp_hidden[i][0].squeeze(2),tmp_cell[i][0].squeeze(2))#.contiguous()
					# The second-layer decoder.
					hidden, states = self.decode_step(
						torch.cat([attention_weighted_encoding, attn_lstm_hidden[0], last_state[0]], dim=1),
						last_state)


					
					hidden = hidden.unsqueeze(0)
					states = states.unsqueeze(0)


					probs_sin = nn.functional.log_softmax(self.output_layer(self.output_projection(hidden)), dim=2) # [1, batch, vocab_size]
					
					# Some modification
					# If <end>
					for e,ww in enumerate(last_word.tolist()):
						if ww[0] == 2:
							probs_sin[:, e, 2] -= 1000  # not going into this beam again!
							# a = done_beams[e,...].T.tolist()
							# b = all_words[g][e,:,i].tolist()
							# if b+[0]*(len(a[0])-len(b)) in a:
							# 	print("debug.")
							if i_e[e] < beam_width*group_size:
								done_beams[e,:all_words[g][e,:,i].shape[0],i_e[e]] = all_words[g][e,:,i] # has done.
								done_beams_sum_prob[e,i_e[e]] = tmp_probs[i][e,0]
								i_e[e] += 1
							# else:
							# 	# check the uniqueness.
							# 	a = done_beams[e,...].T.tolist()
							# 	if len(a) != len([list(x) for x in set(tuple(x) for x in a)]):
							# 		print("SW.")
							# 	# assert len(a) == len([list(x) for x in set(tuple(x) for x in a)]), "Sentences within beam_width should be unique!"
					
					# Diversity enhancement
					# dl = 1000 if t == 0 else diversity_lambda
					for kk in range(g):
						for jj in range(beam_width):
							probs_sin[:, range(self.batch_size),all_words[kk][:,-1, jj]] -= diversity_lambda # for the first word, the words predicted in previous groups cannot show in current one. 
					
					probs, indices = probs_sin.topk(max_num, 2) # change: from bw to hard-core, only top20 is considered, maybe change later! 
					probs = probs.view(self.batch_size, max_num)  # [batch, beam]
					indices = indices.permute(1, 0, 2)  # [batch, 1, beam]
					tmp_words_rep = tmp_words[i].repeat(1, 1, max_num)
					probs_cand = tmp_probs[i] + probs  # [batch, beam]
					words_cand = torch.cat([tmp_words_rep, indices], 1)  # [batch, length+1, beam] screte?
					hidden_cand = hidden.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand = states.unsqueeze(3).repeat(1, 1, 1, max_num)

					hidden_cand_attn = hidden_attn.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand_attn = states_attn.unsqueeze(3).repeat(1, 1, 1, max_num)

					hidden_cand_priorm = hidden_priorm.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand_priorm = states_priorm.unsqueeze(3).repeat(1, 1, 1, max_num)
					hidden_cand_prioro = hidden_prioro.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand_prioro = states_prioro.unsqueeze(3).repeat(1, 1, 1, max_num)
					z_cand_prior = z_t.unsqueeze(3).repeat(1, 1, 1, max_num)

					new_words.append(words_cand)
					new_probs.append(probs_cand)
					new_hidden.append(hidden_cand)
					new_cell.append(cell_cand)
					new_hidden_attn.append(hidden_cand_attn)
					new_cell_attn.append(cell_cand_attn)

					new_hidden_priorm.append(hidden_cand_priorm)
					new_cell_priorm.append(cell_cand_priorm)
					new_hidden_prioro.append(hidden_cand_prioro)
					new_cell_prioro.append(cell_cand_prioro)
					new_z_t_1_prior.append(z_cand_prior)

				new_words = torch.cat(new_words, 2)  # [batch, length+1, beam*mn] expand in the len dim?
				new_probs = torch.cat(new_probs, 1)  # [batch, beam*mn]

				# Diverse BS: diverse augumented probs.
				if g > 0: # first is free to diverse.
					new_probs = self.divaug_probs(new_words, new_probs, all_words)

				new_hidden = torch.cat(new_hidden, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell = torch.cat(new_cell, 3) 
				new_hidden_attn = torch.cat(new_hidden_attn, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell_attn = torch.cat(new_cell_attn, 3) 

				new_hidden_priorm = torch.cat(new_hidden_priorm, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell_priorm = torch.cat(new_cell_priorm, 3)
				new_hidden_prioro = torch.cat(new_hidden_prioro, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell_prioro = torch.cat(new_cell_prioro, 3)
				new_z_t_1_prior = torch.cat(new_z_t_1_prior, 3) 

				# reduce the beam*beam candidates to top@beam candidates
				probs, idx = new_probs.topk(beam_width, 1)  # [batch, beam]
				idx_words = idx.view(self.batch_size, 1, beam_width)
				idx_words = idx_words.repeat(1, t+2, 1)
				idx_states = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.decoder_dim, 1)
				idx_states_attn = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.attention_dim, 1)
				idx_states_priorm = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, 1024, 1) #prior lstm size
				idx_states_prioro = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, 256, 1) #prior lstm size
				idx_states_z_t_1 = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.latent_dim, 1) #z_t_1 size

				# reduce the beam*beam candidates to top@beam candidates
				all_probs[g] = probs
				all_words[g] = new_words.gather(2, idx_words)
				all_hidden[g] = new_hidden.gather(3, idx_states)
				all_cell[g] = new_cell.gather(3, idx_states)
				all_h_attn[g] = new_hidden_attn.gather(3, idx_states_attn)
				all_c_attn[g] = new_cell_attn.gather(3, idx_states_attn)
				all_h_priorm = new_hidden_priorm.gather(3, idx_states_priorm)
				all_c_priorm = new_cell_priorm.gather(3, idx_states_priorm)
				all_h_prioro = new_hidden_prioro.gather(3, idx_states_prioro)
				all_c_prioro = new_cell_prioro.gather(3, idx_states_prioro)
				all_z_t_1 = new_z_t_1_prior.gather(3, idx_states_z_t_1)

				all_z_list.append(all_z_t_1)
				all_h_priorm_list.append(all_h_priorm)
				all_h_prioro_list.append(all_h_prioro)


		idx = idx.view(self.batch_size, beam_width,1, 1).repeat(1,1, max_length+1, 1)
		#captions = all_words.cpu().numpy()
		arginx = torch.argsort(done_beams_sum_prob, dim=1, descending=True)
		# arginx = torch.argsort(probs.cpu(), dim=1, descending=True)
		# captions = torch.cat(all_words, dim=-1).cpu().numpy() # (B, T, b*g)
		db_sort = done_beams.gather(-1, arginx.unsqueeze(1).repeat(1,max_length,1))

		captions = db_sort.cpu().numpy()
		return captions, all_z_list, all_h_priorm_list, all_h_prioro_list

	def divaug_probs(self, new_words, new_probs, hist_seqs):
		'''
		Score with diversity augmentation for Diverse Beam Search
		Refer Paper DBS (AAAI 18).
		Input: 
		-------
		new_words: (B, T, cand_size) # cand_size indicates how many sequences are picked up by the scores.
		new_probs: (B, cand_size) # joint log_prob for each sequence.
		hist_seqs: list [(B, T)]*(g_1) # the decided sequences of previous groups for now.
		
		Output:
		--------
		divaug_new_probs: (B, cand_size) # same size with new_probs, just add some diversity scores.	       
		'''

		return new_probs

	def beam_search2(self, obj_enc, prior_model, max_length=20, beam_width=2, group_size=None, diversity_lambda=None):
		self.batch_size = obj_enc.size(0)
		words = Variable(torch.Tensor([1]).long(), requires_grad=False).repeat(self.batch_size).view(self.batch_size, 1, 1)
		probs = Variable(torch.zeros(self.batch_size, 1))  # [batch, beam]
		if torch.cuda.is_available():
			words = words.cuda()
			probs = probs.cuda()

		h, c = self.init_hidden_state()
		h = h.unsqueeze(0)
		c = c.unsqueeze(0)

		h_attn, c_attn = self.attention.init_attn_lstm_hidden(obj_enc.size(0))
		h_attn = h_attn.unsqueeze(0)
		c_attn = c_attn.unsqueeze(0)

		h_priorm, c_priorm = prior_model.init_attn_lstm_hidden(obj_enc.size(0), 1024)
		h_priorm = h_priorm.unsqueeze(0)
		c_priorm = c_priorm.unsqueeze(0)

		h_prioro, c_prioro = prior_model.init_attn_lstm_hidden(obj_enc.size(0), 256)
		h_prioro = h_prioro.unsqueeze(0)
		c_prioro = c_prioro.unsqueeze(0)

		z_t_1 = torch.zeros(obj_enc.size(0),self.latent_dim).cuda()
		z_t_1 = z_t_1.unsqueeze(0)
		
		all_hidden = h.unsqueeze(3)  # [1, batch, lstm_dim, beam]
		all_cell = c.unsqueeze(3)
		all_h_attn = h_attn.unsqueeze(3)
		all_c_attn = c_attn.unsqueeze(3)
		all_h_priorm = h_priorm.unsqueeze(3)
		all_c_priorm = c_priorm.unsqueeze(3)
		all_h_prioro = h_prioro.unsqueeze(3)
		all_c_prioro = c_prioro.unsqueeze(3)
		all_z_t_1 = z_t_1.unsqueeze(3)


		all_words = words  # [batch, length, beam]
		all_probs = probs  # [batch, beam]
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)
		for t in range(max_length):
			new_words = []
			new_hidden = []
			new_cell = []
			new_hidden_attn = []
			new_cell_attn = []

			new_hidden_priorm = []
			new_cell_priorm = []
			new_hidden_prioro = []
			new_cell_prioro = []
			new_z_t_1_prior = []


			new_probs = []
			#print('t ',t,' all_words ',all_words.size())
			tmp_words = all_words.split(1, 2)
			#print('t ',t,' tmp_words ',len(tmp_words), ' lens ',[c.size() for c in tmp_words])
			tmp_probs = all_probs.split(1, 1)
			#print('t ',t,' all_hidden ',all_hidden.size())
			tmp_hidden = all_hidden.split(1, 3)
			tmp_cell = all_cell.split(1, 3)

			tmp_h_attn = all_h_attn.split(1, 3)
			tmp_c_attn = all_c_attn.split(1, 3)

			tmp_h_priorm = all_h_priorm.split(1, 3)
			tmp_c_priorm = all_c_priorm.split(1, 3)
			tmp_h_prioro = all_h_prioro.split(1, 3)
			tmp_c_prioro = all_c_prioro.split(1, 3)
			tmp_z_t_1 = all_z_t_1.split(1, 3)

			for i in range(len(tmp_words)):
				#print('i_t ',i,' tmp_hidden ',tmp_hidden[0].size())
				last_word = tmp_words[i].split(1, 1)[-1].view(self.batch_size,1)

				last_state_prior = ((tmp_h_priorm[i][0].squeeze(2),tmp_c_priorm[i][0].squeeze(2)),
					(tmp_h_prioro[i][0].squeeze(2),tmp_c_prioro[i][0].squeeze(2)))
				last_z_t_1 = tmp_z_t_1[i][0].squeeze(2)

				z_t, prior_hidden = prior_model(obj_enc = obj_enc, x=last_word, p_z_t_1=tmp_z_t_1[i][0].squeeze(2), 
					hiddens=last_state_prior, q_means_mask=None, q_logs_mask=None, q_z_mask=None, 
					q_means_obj=None, q_logs_obj=None, q_z_obj=None,train=False,reverse=True)

				((hidden_priorm, states_priorm),(hidden_prioro, states_prioro)) = prior_hidden
				hidden_priorm = hidden_priorm.unsqueeze(0)
				states_priorm = states_priorm.unsqueeze(0)
				hidden_prioro = hidden_prioro.unsqueeze(0)
				states_prioro = states_prioro.unsqueeze(0)
				z_t = z_t.unsqueeze(0)

				embeddings = self.embedding(last_word)

				last_state_attn = (tmp_h_attn[i][0].squeeze(2),tmp_c_attn[i][0].squeeze(2))
				attn_lstm_hidden, alpha = self.attention( meanimfeats, z_t.squeeze(0), obj_enc, obj_enc_mask, 
					embeddings[:, 0, :], tmp_hidden[i][0].squeeze(2), last_state_attn)

				hidden_attn, states_attn = attn_lstm_hidden
				hidden_attn = hidden_attn.unsqueeze(0)
				states_attn = states_attn.unsqueeze(0)

				attention_weighted_encoding = (obj_enc * alpha.unsqueeze(2)).sum(dim=1)

				last_state = (tmp_hidden[i][0].squeeze(2),tmp_cell[i][0].squeeze(2))#.contiguous()
				hidden, states = self.decode_step(
					torch.cat([attention_weighted_encoding, attn_lstm_hidden[0], last_state[0]], dim=1),
					last_state)


				
				hidden = hidden.unsqueeze(0)
				states = states.unsqueeze(0)


				probs = nn.functional.log_softmax(self.output_layer(self.output_projection(hidden)), dim=2) # [1, batch, vocab_size]
				probs, indices = probs.topk(beam_width, 2)
				probs = probs.view(self.batch_size, beam_width)  # [batch, beam]
				indices = indices.permute(1, 0, 2)  # [batch, 1, beam]
				tmp_words_rep = tmp_words[i].repeat(1, 1, beam_width)
				probs_cand = tmp_probs[i] + probs  # [batch, beam]
				words_cand = torch.cat([tmp_words_rep, indices], 1)  # [batch, length+1, beam]
				hidden_cand = hidden.unsqueeze(3).repeat(1, 1, 1, beam_width)  # [1, batch, lstm_dim, beam]
				cell_cand = states.unsqueeze(3).repeat(1, 1, 1, beam_width)

				hidden_cand_attn = hidden_attn.unsqueeze(3).repeat(1, 1, 1, beam_width)  # [1, batch, lstm_dim, beam]
				cell_cand_attn = states_attn.unsqueeze(3).repeat(1, 1, 1, beam_width)

				hidden_cand_priorm = hidden_priorm.unsqueeze(3).repeat(1, 1, 1, beam_width)  # [1, batch, lstm_dim, beam]
				cell_cand_priorm = states_priorm.unsqueeze(3).repeat(1, 1, 1, beam_width)
				hidden_cand_prioro = hidden_prioro.unsqueeze(3).repeat(1, 1, 1, beam_width)  # [1, batch, lstm_dim, beam]
				cell_cand_prioro = states_prioro.unsqueeze(3).repeat(1, 1, 1, beam_width)
				z_cand_prior = z_t.unsqueeze(3).repeat(1, 1, 1, beam_width)

				new_words.append(words_cand)
				new_probs.append(probs_cand)
				new_hidden.append(hidden_cand)
				new_cell.append(cell_cand)
				new_hidden_attn.append(hidden_cand_attn)
				new_cell_attn.append(cell_cand_attn)

				new_hidden_priorm.append(hidden_cand_priorm)
				new_cell_priorm.append(cell_cand_priorm)
				new_hidden_prioro.append(hidden_cand_prioro)
				new_cell_prioro.append(cell_cand_prioro)
				new_z_t_1_prior.append(z_cand_prior)

			new_words = torch.cat(new_words, 2)  # [batch, length+1, beam*beam]
			new_probs = torch.cat(new_probs, 1)  # [batch, beam*beam]
			new_hidden = torch.cat(new_hidden, 3)  # [1, batch, lstm_dim, beam*beam]
			new_cell = torch.cat(new_cell, 3) 
			new_hidden_attn = torch.cat(new_hidden_attn, 3)  # [1, batch, lstm_dim, beam*beam]
			new_cell_attn = torch.cat(new_cell_attn, 3) 

			new_hidden_priorm = torch.cat(new_hidden_priorm, 3)  # [1, batch, lstm_dim, beam*beam]
			new_cell_priorm = torch.cat(new_cell_priorm, 3)
			new_hidden_prioro = torch.cat(new_hidden_prioro, 3)  # [1, batch, lstm_dim, beam*beam]
			new_cell_prioro = torch.cat(new_cell_prioro, 3)
			new_z_t_1_prior = torch.cat(new_z_t_1_prior, 3) 

			probs, idx = new_probs.topk(beam_width, 1)  # [batch, beam]
			idx_words = idx.view(self.batch_size, 1, beam_width)
			idx_words = idx_words.repeat(1, t+2, 1)
			idx_states = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.decoder_dim, 1)
			idx_states_attn = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.attention_dim, 1)
			idx_states_priorm = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, 1024, 1) #prior lstm size
			idx_states_prioro = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, 256, 1) #prior lstm size
			idx_states_z_t_1 = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.latent_dim, 1) #z_t_1 size

			# reduce the beam*beam candidates to top@beam candidates
			all_probs = probs
			all_words = new_words.gather(2, idx_words)
			all_hidden = new_hidden.gather(3, idx_states)
			all_cell = new_cell.gather(3, idx_states)
			all_h_attn = new_hidden_attn.gather(3, idx_states_attn)
			all_c_attn = new_cell_attn.gather(3, idx_states_attn)
			all_h_priorm = new_hidden_priorm.gather(3, idx_states_priorm)
			all_c_priorm = new_cell_priorm.gather(3, idx_states_priorm)
			all_h_prioro = new_hidden_prioro.gather(3, idx_states_prioro)
			all_c_prioro = new_cell_prioro.gather(3, idx_states_prioro)
			all_z_t_1 = new_z_t_1_prior.gather(3, idx_states_z_t_1)

		idx = idx.view(self.batch_size, beam_width,1, 1).repeat(1,1, max_length+1, 1)
		captions = all_words.cpu().numpy() 
		return captions, None, None, None
	
	# try to reproduce AAAI BS
	def beam_search3(self, obj_enc, prior_model, max_length=20, beam_width=2, group_size=1, diversity_lambda=0.8, post_z=None):
		max_num = max(5, beam_width)
		self.batch_size = obj_enc.size(0)
		# index 1 is <SOS>
		words = Variable(torch.Tensor([1]).long(), requires_grad=False).repeat(self.batch_size).view(self.batch_size, 1, 1)
		probs = Variable(torch.zeros(self.batch_size, 1))  # [batch, beam]
		if torch.cuda.is_available():
			words = words.cuda()
			probs = probs.cuda()

		h, c = self.init_hidden_state()
		h = h.unsqueeze(0)
		c = c.unsqueeze(0)

		h_attn, c_attn = self.attention.init_attn_lstm_hidden(obj_enc.size(0))
		h_attn = h_attn.unsqueeze(0)
		c_attn = c_attn.unsqueeze(0)

		h_priorm, c_priorm = prior_model.init_attn_lstm_hidden(obj_enc.size(0), 1024)
		h_priorm = h_priorm.unsqueeze(0)
		c_priorm = c_priorm.unsqueeze(0)

		h_prioro, c_prioro = prior_model.init_attn_lstm_hidden(obj_enc.size(0), 256)
		h_prioro = h_prioro.unsqueeze(0)
		c_prioro = c_prioro.unsqueeze(0)

		z_t_1 = torch.zeros(obj_enc.size(0),128).cuda()
		z_t_1 = z_t_1.unsqueeze(0)
		
		all_hidden = [h.unsqueeze(3) for _ in range(group_size)]  # [1, batch, lstm_dim, beam]
		all_cell = [c.unsqueeze(3) for _ in range(group_size)]
		all_h_attn = [h_attn.unsqueeze(3) for _ in range(group_size)]
		all_c_attn = [c_attn.unsqueeze(3) for _ in range(group_size)]
		all_h_priorm = h_priorm.unsqueeze(3)
		all_c_priorm = c_priorm.unsqueeze(3) 
		all_h_prioro = h_prioro.unsqueeze(3)
		all_c_prioro = c_prioro.unsqueeze(3)
		all_z_t_1 = z_t_1.unsqueeze(3)

		all_z_list = []
		all_h_priorm_list = []
		all_h_prioro_list = []

		all_words = [words for _ in range(group_size)]  # [batch, length, beam] # how to extend in the length dim?
		all_probs = [probs for _ in range(group_size)] # [batch, beam]
		obj_enc_mask = torch.sum(torch.abs(obj_enc), dim=-1) > 0
		meanimfeats = masked_mean( obj_enc, obj_enc_mask.unsqueeze(-1) ,dim=1)

		done_beams = torch.zeros(self.batch_size, max_length, beam_width*group_size, dtype=int)
		done_beams_sum_prob = torch.zeros(self.batch_size, beam_width*group_size)
		i_e = [0 for _ in range(self.batch_size)]

		for t in range(max_length-1):

			for g in range(group_size):
				# Diverse Beam Search, 
				# if group_size == 1, it is equal to BS.
				new_words = []
				new_hidden = []
				new_cell = []
				new_hidden_attn = []
				new_cell_attn = []

				new_hidden_priorm = []
				new_cell_priorm = []
				new_hidden_prioro = []
				new_cell_prioro = []
				new_z_t_1_prior = []


				new_probs = []
				# self.logger.debug('t ',t,' all_words ',all_words.size())
				tmp_words = all_words[g].split(1, 2)
				# self.logger.debug('t ',t,' tmp_words ',len(tmp_words), ' lens ',[c.size() for c in tmp_words])
				tmp_probs = all_probs[g].split(1, 1)
				# self.logger.debug('t ',t,' all_hidden ',all_hidden.size())
				tmp_hidden = all_hidden[g].split(1, 3)
				tmp_cell = all_cell[g].split(1, 3)

				tmp_h_attn = all_h_attn[g].split(1, 3)
				tmp_c_attn = all_c_attn[g].split(1, 3)

				tmp_h_priorm = all_h_priorm.split(1, 3)
				tmp_c_priorm = all_c_priorm.split(1, 3)
				tmp_h_prioro = all_h_prioro.split(1, 3)
				tmp_c_prioro = all_c_prioro.split(1, 3)
				tmp_z_t_1 = all_z_t_1.split(1, 3)

				for i in range(len(tmp_words)):
					# self.logger.debug('This is the time %d', t) 
					# self.logger.debug('This is the tmp word %d', i)
					#print('i_t ',i,' tmp_hidden ',tmp_hidden[0].size())
					last_word = tmp_words[i].split(1, 1)[-1].view(self.batch_size,1)

					last_state_prior = ((tmp_h_priorm[i][0].squeeze(2),tmp_c_priorm[i][0].squeeze(2)),
						(tmp_h_prioro[i][0].squeeze(2),tmp_c_prioro[i][0].squeeze(2)))
					last_z_t_1 = tmp_z_t_1[i][0].squeeze(2)

					# DEBUG: print out the z_t in different sample process, check out how it varies
					z_t, prior_hidden = prior_model(obj_enc = obj_enc, x=last_word, p_z_t_1=tmp_z_t_1[i][0].squeeze(2), 
						hiddens=last_state_prior, q_means_mask=None, q_logs_mask=None, q_z_mask=None, 
						q_means_obj=None, q_logs_obj=None, q_z_obj=None,train=False,reverse=True)

					((hidden_priorm, states_priorm),(hidden_prioro, states_prioro)) = prior_hidden
					hidden_priorm = hidden_priorm.unsqueeze(0)
					states_priorm = states_priorm.unsqueeze(0)
					hidden_prioro = hidden_prioro.unsqueeze(0)
					states_prioro = states_prioro.unsqueeze(0)
					z_t = z_t.unsqueeze(0)
					# if self.debug:
					# r1, r2 = 0, 10
					# z_t = torch.FloatTensor(z_t.shape).normal_(r1, r2).cuda()
					# 	assert z_t.shape[1] == 1, "in the debug stage, the batch size should be one." # BS == 1
					# 	# self.logger.debug("Time step: {}".format(t))
					# 	# self.logger.debug("Tmp words: {}".format(tmp_words))
					# 	# self.logger.debug("Sampled z_t: {}".format(z_t[0,0,:]))
						# self.logger.debug("The last word is %d", last_word[0].item())
						# self.logger.debug(torch.mean(z_t[0]))


					embeddings = self.embedding(last_word)

					last_state_attn = (tmp_h_attn[i][0].squeeze(2),tmp_c_attn[i][0].squeeze(2))
					# The first-layer decoder.
					attn_lstm_hidden, alpha = self.attention( meanimfeats, z_t.squeeze(0), obj_enc, obj_enc_mask, 
						embeddings[:, 0, :], tmp_hidden[i][0].squeeze(2), last_state_attn)

					hidden_attn, states_attn = attn_lstm_hidden
					hidden_attn = hidden_attn.unsqueeze(0)
					states_attn = states_attn.unsqueeze(0)

					attention_weighted_encoding = (obj_enc * alpha.unsqueeze(2)).sum(dim=1)

					last_state = (tmp_hidden[i][0].squeeze(2),tmp_cell[i][0].squeeze(2))#.contiguous()
					# The second-layer decoder.
					hidden, states = self.decode_step(
						torch.cat([attention_weighted_encoding, attn_lstm_hidden[0], last_state[0]], dim=1),
						last_state)


					
					hidden = hidden.unsqueeze(0)
					states = states.unsqueeze(0)


					probs_sin = nn.functional.log_softmax(self.output_layer(self.output_projection(hidden)), dim=2) # [1, batch, vocab_size]
					
					# Some modification
					# If <end>
					for e,ww in enumerate(last_word.tolist()):
						if ww[0] == 2 and done_beams[e, -1, g] != -1:
							probs_sin[:, e, 2] -= 1000  # not going into this beam again!
							# a = done_beams[e,...].T.tolist()
							# b = all_words[g][e,:,i].tolist()
							# if b+[0]*(len(a[0])-len(b)) in a:
							# 	print("debug.")
							if i_e[e] < beam_width*group_size:
								done_beams[e,:all_words[g][e,:,i].shape[0],i_e[e]] = all_words[g][e,:,i] # has done.
								done_beams_sum_prob[e,i_e[e]] = tmp_probs[i][e,0]
								done_beams[e, -1, g] = -1 # set the end symbol
								i_e[e] += 1
							# else:
							# 	# check the uniqueness.
							# 	a = done_beams[e,...].T.tolist()
							# 	if len(a) != len([list(x) for x in set(tuple(x) for x in a)]):
							# 		print("SW.")
							# 	# assert len(a) == len([list(x) for x in set(tuple(x) for x in a)]), "Sentences within beam_width should be unique!"
					
					# Diversity enhancement
					# dl = 1000 if t == 0 else diversity_lambda
					for kk in range(g):
						for jj in range(beam_width):
							probs_sin[:, range(self.batch_size),all_words[kk][:,-1, jj]] -= diversity_lambda # for the first word, the words predicted in previous groups cannot show in current one. 
					
					probs, indices = probs_sin.topk(max_num, 2) # change: from bw to hard-core, only top20 is considered, maybe change later! 
					probs = probs.view(self.batch_size, max_num)  # [batch, beam]
					indices = indices.permute(1, 0, 2)  # [batch, 1, beam]
					tmp_words_rep = tmp_words[i].repeat(1, 1, max_num)
					probs_cand = tmp_probs[i] + probs  # [batch, beam]
					words_cand = torch.cat([tmp_words_rep, indices], 1)  # [batch, length+1, beam] screte?
					hidden_cand = hidden.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand = states.unsqueeze(3).repeat(1, 1, 1, max_num)

					hidden_cand_attn = hidden_attn.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand_attn = states_attn.unsqueeze(3).repeat(1, 1, 1, max_num)

					hidden_cand_priorm = hidden_priorm.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand_priorm = states_priorm.unsqueeze(3).repeat(1, 1, 1, max_num)
					hidden_cand_prioro = hidden_prioro.unsqueeze(3).repeat(1, 1, 1, max_num)  # [1, batch, lstm_dim, beam]
					cell_cand_prioro = states_prioro.unsqueeze(3).repeat(1, 1, 1, max_num)
					z_cand_prior = z_t.unsqueeze(3).repeat(1, 1, 1, max_num)

					new_words.append(words_cand)
					new_probs.append(probs_cand)
					new_hidden.append(hidden_cand)
					new_cell.append(cell_cand)
					new_hidden_attn.append(hidden_cand_attn)
					new_cell_attn.append(cell_cand_attn)

					new_hidden_priorm.append(hidden_cand_priorm)
					new_cell_priorm.append(cell_cand_priorm)
					new_hidden_prioro.append(hidden_cand_prioro)
					new_cell_prioro.append(cell_cand_prioro)
					new_z_t_1_prior.append(z_cand_prior)

				new_words = torch.cat(new_words, 2)  # [batch, length+1, beam*mn] expand in the len dim?
				new_probs = torch.cat(new_probs, 1)  # [batch, beam*mn]

				# Diverse BS: diverse augumented probs.
				if g > 0: # first is free to diverse.
					new_probs = self.divaug_probs(new_words, new_probs, all_words)

				new_hidden = torch.cat(new_hidden, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell = torch.cat(new_cell, 3) 
				new_hidden_attn = torch.cat(new_hidden_attn, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell_attn = torch.cat(new_cell_attn, 3) 

				new_hidden_priorm = torch.cat(new_hidden_priorm, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell_priorm = torch.cat(new_cell_priorm, 3)
				new_hidden_prioro = torch.cat(new_hidden_prioro, 3)  # [1, batch, lstm_dim, beam*mn]
				new_cell_prioro = torch.cat(new_cell_prioro, 3)
				new_z_t_1_prior = torch.cat(new_z_t_1_prior, 3) 

				# reduce the beam*beam candidates to top@beam candidates
				probs, idx = new_probs.topk(beam_width, 1)  # [batch, beam]
				idx_words = idx.view(self.batch_size, 1, beam_width)
				idx_words = idx_words.repeat(1, t+2, 1)
				idx_states = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.decoder_dim, 1)
				idx_states_attn = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, self.attention_dim, 1)
				idx_states_priorm = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, 1024, 1) #prior lstm size
				idx_states_prioro = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, 256, 1) #prior lstm size
				idx_states_z_t_1 = idx.view(1, self.batch_size, 1, beam_width).repeat(1, 1, 128, 1) #z_t_1 size

				# reduce the beam*beam candidates to top@beam candidates
				all_probs[g] = probs
				all_words[g] = new_words.gather(2, idx_words)
				all_hidden[g] = new_hidden.gather(3, idx_states)
				all_cell[g] = new_cell.gather(3, idx_states)
				all_h_attn[g] = new_hidden_attn.gather(3, idx_states_attn)
				all_c_attn[g] = new_cell_attn.gather(3, idx_states_attn)
				all_h_priorm = new_hidden_priorm.gather(3, idx_states_priorm)
				all_c_priorm = new_cell_priorm.gather(3, idx_states_priorm)
				all_h_prioro = new_hidden_prioro.gather(3, idx_states_prioro)
				all_c_prioro = new_cell_prioro.gather(3, idx_states_prioro)
				all_z_t_1 = new_z_t_1_prior.gather(3, idx_states_z_t_1)

				all_z_list.append(all_z_t_1)
				all_h_priorm_list.append(all_h_priorm)
				all_h_prioro_list.append(all_h_prioro)


		idx = idx.view(self.batch_size, beam_width,1, 1).repeat(1,1, max_length+1, 1)
		#captions = all_words.cpu().numpy()
		arginx = torch.argsort(done_beams_sum_prob, dim=1, descending=True)
		# captions = torch.cat(all_words, dim=-1).cpu().numpy() # (B, T, b*g)
		db_sort = done_beams.gather(-1, arginx.unsqueeze(1).repeat(1,max_length,1))

		captions = db_sort.cpu().numpy()
		return captions, all_z_list, all_h_priorm_list, all_h_prioro_list
	
	def forward(self, seq, meanimfeats, sent_enc, obj_enc, caption_lengths, bw, gs=1, dl=0.2, prior_model=None, train = True, post_z=None):
		if train:
			return self._train(seq, meanimfeats, sent_enc, obj_enc, caption_lengths, prior_model)
		else:
			return self.beam_search(obj_enc, prior_model, max_length=caption_lengths, beam_width=bw, group_size=gs, diversity_lambda=dl, post_z=post_z)

	
