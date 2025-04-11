import torch
from utils import load_configs
from model import LNN, RNN


# Load configuration
configs = load_configs('config.yaml')

# Device
device = torch.device(configs['device'])
print(f"device: {device}")


if __name__ == '__main__':
    # Model parameters
    model_cfg = configs['model']
    cell = model_cfg['cell']









# import torch
# import torch.nn as nn
# from utils import load_configs
# from modules import LTCCell, CfCCell, CfCImprovedCell
# from model import LNNEnsemble, LNN
# from dataset import get_sequential_MNIST_dataloaders, get_sequential_transforms
# from train import Trainer


# # Load configuration
# configs = load_configs('config.yaml')

# # Device
# device = torch.device(configs['device'])
# print(f'device: {device}')

# if device.type == 'cuda':
#     print(f'cuda available: {torch.cuda.is_available()}')


# if __name__ == '__main__':
#     # Model parameters
#     model_cfgs = configs['model']
#     cell = model_cfgs['cell']
#     in_features = model_cfgs['in_features']
#     hidden_features = model_cfgs['hidden_features']
#     out_features = model_cfgs['out_features']
#     backbone_features = model_cfgs['backbone_features']
#     backbone_depth = model_cfgs['backbone_depth']

#     # Ensemble parameters
#     ensemble_cfgs = configs['ensemble']
#     enable_ensemble = ensemble_cfgs['enable_ensemble']
#     ensemble_size = ensemble_cfgs['ensemble_size']

#     # Training parameters
#     train_cfgs = configs['train']
#     loss_fn = train_cfgs['loss_fn']
#     epochs = train_cfgs['epochs']
#     batch_size = train_cfgs['batch_size']
#     lr = train_cfgs['lr']
#     tau = train_cfgs['tau']


#     # Cell selection
#     if cell == 'LTCCell':
#         cell = LTCCell
#     elif cell == 'CfCCell':
#         cell = CfCCell
#     elif cell == 'CfCImprovedCell':
#         cell = CfCImprovedCell

    
#     # Loss function selection
#     if loss_fn == 'CE':
#         loss_fn = nn.CrossEntropyLoss()
#     elif loss_fn == 'MSE':
#         loss_fn = nn.MSELoss()
#     elif loss_fn == 'BCE':
#         loss_fn = nn.BCELoss()


#     # Model
#     if enable_ensemble:
#         model = LNNEnsemble(
#             base_model=lambda: LNN(
#                 cell=cell,
#                 in_features=in_features,
#                 hidden_features=hidden_features,
#                 out_features=out_features,
#                 backbone_features=backbone_features,
#                 backbone_depth=backbone_depth
#             ),
#             ensemble_size=ensemble_size
#         )
#     else:
#         model = LNN(
#             cell=cell,
#             in_features=in_features,
#             hidden_features=hidden_features,
#             out_features=out_features,
#             backbone_features=backbone_features,
#             backbone_depth=backbone_depth
#         )
#     parameters = model._get_parameters()
#     print(f"Number of parameters: {parameters}")


#     # Dataset loaders
#     train_loader, test_loader = get_sequential_MNIST_dataloaders(batch_size=batch_size, transform=get_sequential_transforms())


#     # Trainr
#     trainer = Trainer(
#         device=device,
#         model=model,
#         loss_fn=loss_fn,
#         train_loader=train_loader,
#         test_loader=test_loader,
#         epochs=epochs,
#         batch_size=batch_size,
#         lr=lr,
#         tau=tau
#     )
#     trainer.fit()
#     trainer._save_model(path='model.pth')

