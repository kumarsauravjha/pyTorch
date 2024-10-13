#%%
import torch
import torch.nn as nn
device = ('cuda' if torch.cuda.is_available() else 'cpu')


# %%
new_model = torch.jit.load('iris_nn_3.pt')
# %%
new_model.eval()
# new_model.to(device)
#%%

#%%
new = torch.tensor([7.0, 5.0, 7.0, 5.0]).to(device)

with torch.no_grad():
    pred = new_model.forward(new)
    print(pred.argmax().item())
# %%
