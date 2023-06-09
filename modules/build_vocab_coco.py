import nltk
import pickle
import argparse
from collections import Counter
import sys
from tqdm import tqdm
import re

sys.path.append('cococaption')
from pycocotools.coco import COCO
# nltk.download('punkt')
import json



class Vocabulary(object):
	"""Simple vocabulary wrapper."""
	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	def add_word(self, word):
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	def __call__(self, word):
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)

def build_vocab(json_file, threshold):
	"""Build a simple vocabulary wrapper."""
	coco = COCO(json_file)
	counter = Counter()
	ids = coco.anns.keys()
	for i, id in tqdm(enumerate(ids)):
		caption = str(coco.anns[id]['caption'])
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		counter.update(tokens)

		# if (i+1) % 1000 == 0:
		# 	print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

	# If the word frequency is less than 'threshold', then the word is discarded.
	words1 = [word for word, cnt in counter.items() if cnt >= threshold]
 
	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()
	for w in words1:
		vocab.add_word(w)
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')
	vocab.add_word('<temp>')


	# Add the words to the vocabulary.
	# for i, word in enumerate(words):
	# 	vocab.add_word(word)
	# category_reader = json.load(open('./data/oi_categories.json'))
	# for rows in category_reader:
	# 	vocab.add_word(str(rows["name"]).lower())

	# print(vocab)
	return vocab

def main(args):
	vocab = build_vocab(json_file=args.caption_path, threshold=args.threshold)

	vocab_path = args.vocab_path
	with open(vocab_path, 'w') as f:
		# pickle.dump(vocab, f)
		cand = list(vocab.word2idx.keys())
		cand = [i for i in cand if re.compile(r'[a-z]').match(i)]
		json.dump(cand, f)
	print("Total vocabulary size: {}".format(len(vocab)))
	print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--caption_path', type=str, 
						default="datasets/coco/coco_train_cocofmt.json", 
						help='path for train annotation file')
	parser.add_argument('--vocab_path', type=str, default='./datasets/coco/coco_vocab.json', 
						help='path for saving vocabulary wrapper')
	parser.add_argument('--threshold', type=int, default=4, 
						help='minimum word count threshold')
	args = parser.parse_args()
	main(args)