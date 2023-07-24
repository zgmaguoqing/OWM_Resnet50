import torch, sys

print('Run success!')

if torch.cuda.is_available():
    print('[CUDA available]')
    device = 'cuda'
else:
    print('[CUDA unavailable]')
    sys.exit()

a = torch.rand((2, 3)).to(device)
print(a)