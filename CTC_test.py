import torch
import torch.nn.functional as F


logits = torch.randn(10, 32, 20)  # time setp: 10, batch: 32, class: 20
ctc_decoder = F.ctc_loss


char_map = ['-', 'A', 'B', 'C', '1', '2', '3', '4', '5']
target_labels = [1, 2, 3, 4, 5]  