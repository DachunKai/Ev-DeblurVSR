import torch
import numbers
import torch.nn as nn
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class MultiHeadAttn(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):

        b,c,h,w = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CrossAttn(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Mlp(nn.Module):
    def __init__(self, dim=64, out_dim=64, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim,
                                              kernel_size=3, stride=1, padding=1),
                                    act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, input_size):
        # bs x hw x c
        B, L, C = x.size()
        H, W = input_size
        assert H * W == L, "output H x W is not the same with L!"

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)  # bs, hidden_dim, 32x32
        x = self.dwconv(x)

        # flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)

        return x


class DualChannelBlock(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=4,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()

        self.image_norm1 = LayerNorm(dim, LayerNorm_type)
        self.event_norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_image = MultiHeadAttn(dim, num_heads, bias)
        self.attn_event = MultiHeadAttn(dim, num_heads, bias)
        self.image_norm2 = LayerNorm(dim, LayerNorm_type)
        self.event_norm2 = LayerNorm(dim, LayerNorm_type)
        self.cross_attn_img = CrossAttn(dim, num_heads, bias)
        self.cross_attn_event = CrossAttn(dim, num_heads, bias)
        self.event_norm3 = LayerNorm(dim, LayerNorm_type)

        # mlp
        self.image_norm3 = nn.LayerNorm(dim)
        self.event_norm4 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.image_ffn = Mlp(dim=dim, out_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.event_ffn = Mlp(dim=dim, out_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c, h, w = image.shape

        # self attention
        image = self.image_norm1(image)
        event = self.event_norm1(event)
        image = image + self.attn_image(image)
        event = event + self.attn_event(event)

        # cross attention
        image = self.image_norm2(image)
        event = self.event_norm2(event)
        event = event + self.cross_attn_event(event, image)
        event = self.event_norm3(event)
        image = image + self.cross_attn_img(image, event)

        # mlp
        image = to_3d(image) # b, h*w, c
        image = self.image_norm3(image)
        image = image + self.image_ffn(image, (h, w))
        image = to_4d(image, h, w)

        event = to_3d(event) # b, h*w, c
        event = self.event_norm4(event)
        event = event + self.event_ffn(event, (h, w))
        event = to_4d(event, h, w)

        return image, event