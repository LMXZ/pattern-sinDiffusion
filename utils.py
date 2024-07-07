import torch
import torch.nn.functional as F

class AvgMeter:
    def __init__(self, name: str, interval: int, mode: str='window') -> None:
        self.mode = mode
        self.sum = 0
        self.cnt = 0
        self.interval = interval
        self.name = name
    
    def add(self, x):
        self.sum += x
        self.cnt += 1
        if self.cnt % self.interval == self.interval - 1:
            print(f'{self.name}: {self.sum / self.cnt}')
            if self.mode == 'window':
                self.clear()
    
    def clear(self):
        self.sum = 0
        self.cnt = 0

def simi_x(full_image, h, w):
    piece = full_image[:, :h, :w]
    return F.mse_loss(piece.repeat(1, 2, 1), full_image[:, :h*2, :w])

def simi_y(full_image, h, w):
    piece = full_image[:, :h, :w]
    return F.mse_loss(piece.repeat(1, 1, 2), full_image[:, :h, :w*2])

def find_piece_size(full_image, begin, end): # 找一个合适的块大小
    simi_xs = torch.stack([simi_x(full_image, i, 50) for i in range(begin, end)])
    simi_ys = torch.stack([simi_y(full_image, 50, i) for i in range(begin, end)])
    h, w = int(simi_xs.argmin() + begin), int(simi_ys.argmin() + begin)
    return h, w