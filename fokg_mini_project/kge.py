from abc import ABC, abstractmethod
import torch
from fokg_mini_project.kg import KG
from fokg_mini_project.dataset import Dataset
from tqdm import tqdm
from typing import Callable
import random

class ForwardFunction(ABC):
    @abstractmethod
    def __call__(self, emb_ent, emb_rel, e1_idx, rel_idx, e2_idx, *args, **kwargs):
        """Defines the expected structure of the forward function."""
        pass

class LossFunction(ABC):
    @abstractmethod
    def __call__(self, os_scores, neg_scores, *args, **kwargs):
        """Defines the expected structure of the loss function."""
        pass


class KGE(torch.nn.Module, ABC):
    def __init__(self, kb: KG,embedding_dim = 100, optimizer = None, device='cuda', lr=0.001, forward: Callable = None, loss_func: Callable = None):
        super(KGE, self).__init__()
        self.forward = forward
        self.loss_func = loss_func
        self.model_name = forward.__class__.__name__
        self.kb = kb
        self.embedding_dim = embedding_dim 
        from fokg_mini_project.complex import ComplexForward
        if isinstance(forward, ComplexForward):
            print("Complex model")
            self.embedding_dim = embedding_dim * 2
        self.num_entities = kb.num_entities
        self.num_relations = kb.num_relations

        self.emb_ent = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.emb_rel = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        
        self.low = -6/torch.sqrt(torch.tensor(self.embedding_dim)).item()
        self.high = 6/torch.sqrt(torch.tensor(self.embedding_dim)).item()
        
        
        self.emb_ent.weight.data.uniform_(self.low, self.high)
        self.emb_rel.weight.data.uniform_(self.low, self.high)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        if optimizer is not None:
            self.optimizer = optimizer

        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def train(self, num_epochs: int = 100, batch_size = 1024):
        dataloader = self.get_dataloader(batch_size)
        for epoch in range(0, num_epochs):
            runnning_loss = 0.0
            for h, r, t, labels in tqdm(dataloader, desc=f"Training looop"):
                h, r, t, labels = h.to(self.device), r.to(self.device), t.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                midpoint = labels.size(0) // 2
                pos_distance = self.forward(self.emb_ent, self.emb_rel,h[:midpoint], r[:midpoint], t[:midpoint])
                neg_distance = self.forward(self.emb_ent, self.emb_rel,h[midpoint:], r[midpoint:], t[midpoint:])
                loss = self.loss_func(pos_distance, neg_distance)
                runnning_loss += loss.item() 
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {runnning_loss/len(dataloader.dataset)}")

    def save_model(self, path=None):
        if path is None:
            torch.save(self.state_dict(), self.model_name.replace("Forward", "_") + self.kb.kb_path.split("/")[-1].replace(".nt", ".pth"))
        else:
            if not path.endswith(".pth"):
                path += ".pth"
            torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_dataloader(self, batch_size):
        dataset_train = self.get_dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            collate_fn=dataset_train.collate_fn
        )
        return dataloader

    def get_dataset(self):
        dataset = Dataset(self.kb.triples_idx, self.kb.num_entities)
        return dataset

    def score_triple(self, head_idx, rel_idx, tail_idx):
        head_idx = torch.tensor([head_idx], dtype=torch.long, device=self.device)
        rel_idx = torch.tensor([rel_idx], dtype=torch.long, device=self.device)
        tail_idx = torch.tensor([tail_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            distance = self.forward(self.emb_ent, self.emb_rel, head_idx, rel_idx, tail_idx)
        return distance.item()

    def score_triple_normalized(self, head_idx, rel_idx, tail_idx):
        head_idx = torch.tensor([head_idx], dtype=torch.long, device=self.device)
        rel_idx = torch.tensor([rel_idx], dtype=torch.long, device=self.device)
        tail_idx = torch.tensor([tail_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            distance = self.forward(self.emb_ent, self.emb_rel, head_idx, rel_idx, tail_idx)
        if self.model_name == "ComplexForward":
            return torch.sigmoid(-distance).item()

    def predict_tail(self, head_idx, rel_idx, top_k=5):
        """
        Given a head and a relation, predict the top_k candidate tails.
        Returns a list of (entity_idx, distance) pairs, sorted by distance.
        """
        # Create a batch of all possible tails
        all_tails = torch.arange(self.kb.num_entities, device=self.device)
        head_vec = torch.full((self.kb.num_entities,), head_idx, dtype=torch.long, device=self.device)
        rel_vec = torch.full((self.kb.num_entities,), rel_idx, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            distances = self.forward(self.emb_ent, self.emb_rel, head_vec, rel_vec, all_tails)
        # Sort by distance (ascending - smaller is better)
        sorted_distances, sorted_indices = torch.sort(distances)
        # Retrieve top_k tails
        top_tails = [(sorted_indices[i].item(), sorted_distances[i].item()) for i in range(top_k)]
        return top_tails

    def predict_head(self, tail_idx, rel_idx, top_k=5):
        """
        Given a tail and a relation, predict the top_k candidate heads.
        Similar logic to predict_tail.
        """
        all_heads = torch.arange(self.kb.num_entities, device=self.device)
        tail_vec = torch.full((self.kb.num_entities,), tail_idx, dtype=torch.long, device=self.device)
        rel_vec = torch.full((self.kb.num_entities,), rel_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Note the order: head is the candidate, so we put it as the first arg in forward,
            # the relation next, and the known tail last.
            distances = self.forward(self.emb_ent, self.emb_rel, all_heads, rel_vec, tail_vec)
        # Sort the scores
        sorted_distances, sorted_indices = torch.sort(distances)
        top_heads = [(sorted_indices[i].item(), sorted_distances[i].item()) for i in range(top_k)]
        return top_heads
    
    def predict_relation(self, head_idx, tail_idx, top_k=10):
        """
        Given a head and a tail, predict the top_k candidate relations.
        """
        all_rels = torch.arange(self.kb.num_relations, device=self.device)
        head_vec = torch.full((self.kb.num_relations,), head_idx, dtype=torch.long, device=self.device)
        tail_vec = torch.full((self.kb.num_relations,), tail_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            distances = self.forward(self.emb_ent, self.emb_rel, head_vec, all_rels, tail_vec)
        # Sort and retrieve the top_k
        sorted_distances, sorted_indices = torch.sort(distances)
        top_rels = [(sorted_indices[i].item(), sorted_distances[i].item()) for i in range(top_k)]
        return top_rels

    def eval_hits(self, hits_at=10, sample_ratio = 0.001):
        hits = 0
        sample_triples = random.sample(self.kb.triples_idx, round(len(self.kb.triples_idx) * sample_ratio))
        for h,r,t in tqdm(sample_triples, desc="Evaluating"):
            for t̅, _ in self.predict_tail(h,r,hits_at):
                if t == t̅:
                    hits += 1
        return hits/len(sample_triples)

            

