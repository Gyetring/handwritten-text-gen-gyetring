from dataloader import get_dataloader

loader = get_dataloader(batch_size=8)

for x, y in loader:
    print(x.shape, y.shape)
    break
