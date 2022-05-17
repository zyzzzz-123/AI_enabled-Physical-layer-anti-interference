import torch

# pytorch的标记默认从0开始
class_num = 2
batch_size = 4
label = torch.LongTensor(batch_size, 1).random_() % class_num
a = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
print(label)
print(a)
