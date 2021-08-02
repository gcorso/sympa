import torch
import pickle

s = torch.load('./ckpt/11-best-3000ep')
pickle.dump(s['model']['module.embeddings.embeds'].float(), open('./emb/ba_HYPS16_882.pkl', "wb" ))