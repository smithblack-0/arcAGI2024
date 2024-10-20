import torch



insert = torch.randn([3, 5, ])
embed = torch.diag_embed(insert, 0, -2, -1)
print(embed.shape)