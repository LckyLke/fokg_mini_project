import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, num_entities=None):
        data = torch.Tensor(data).long()
        self.head_idx = data[:, 0]
        self.rel_idx = data[:, 1]
        self.tail_idx = data[:, 2]
        self.num_entities = num_entities
        assert self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape

        self.length = len(self.head_idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h = self.head_idx[idx]
        r = self.rel_idx[idx]
        t = self.tail_idx[idx]
        return h, r, t

    def collate_fn(self, batch):
        """ Generate Negative Triples"""
        batch = torch.LongTensor(batch)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        size_of_batch, _ = batch.shape
        assert size_of_batch > 0
        label = torch.ones((size_of_batch, ))
        corr = torch.randint(0, self.num_entities, (size_of_batch, 1))
        
        if torch.rand(1).item() > 0.5:
            h_corr = corr[:, 0]
            r_corr = r
            t_corr = t
            label_corr = -torch.ones(size_of_batch, )
        else:
            h_corr = h
            r_corr = r
            t_corr = corr[:, 0]
            label_corr = -torch.ones(size_of_batch, )

        h = torch.cat((h, h_corr), 0)
        r = torch.cat((r, r_corr), 0)
        t = torch.cat((t, t_corr), 0)
        label = torch.cat((label, label_corr), 0)
        return h, r, t, label