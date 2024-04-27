import torch 
data = torch.load(weibo.pth)
data['corpus'] = data['corpus'][:106478]
torch.save(data, 'weibo_small.pth')
