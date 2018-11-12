import torch
import sys


def get_tensor_bytes(tensor):
    element_num = torch.prod(torch.tensor(tensor.size()))
    size_mapping = {
        torch.int:4,
        torch.long:8,
        torch.float:4
    }
    return (size_mapping[tensor.dtype] * element_num).item()


def resize_bbox(bboxes, scale_factor):
    bboxes = bboxes.float()
    bboxes.mul_(scale_factor)
    return bboxes.int()


if __name__ == '__main__':
    print(get_tensor_bytes(torch.tensor([1, 2])))