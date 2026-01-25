import torch
import torch.nn.functional as F


def attention_improvement_loss(node_embeddings, attention_embeddings, edge_index, batch, tau=0.2):
    """
    Compute attention improvement loss to encourage better structural associations.
    
    Args:
        node_embeddings: Initial node embeddings [N, D]
        attention_embeddings: Embeddings after attention transformation [N, D]
        edge_index: Edge indices [2, E]
        batch: Batch assignment for each node [N]
        tau: Temperature for sigmoid
    """
    # Normalize embeddings
    node_emb_normed = F.normalize(node_embeddings, p=2, dim=-1)
    attn_emb_normed = F.normalize(attention_embeddings, p=2, dim=-1)
    
    src, dst = edge_index
    
    # Compute similarity scores for connected nodes
    initial_scores = (node_emb_normed[src] * node_emb_normed[dst]).sum(-1)
    final_scores = (attn_emb_normed[src] * attn_emb_normed[dst]).sum(-1)
    
    # Compute per-graph threshold as mean initial score
    edge_batch = batch[src]  # batch assignment for each edge
    num_graphs = batch.max().item() + 1
    
    with torch.no_grad():
        threshold = torch.zeros_like(initial_scores)
        for b in range(num_graphs):
            mask = (edge_batch == b)
            if mask.sum() > 0:
                threshold[mask] = initial_scores[mask].mean()
    
    # Soft recall using sigmoid
    recall_final = torch.sigmoid((final_scores - threshold) / tau)
    
    # Compute per-graph loss
    per_graph_loss = torch.zeros(num_graphs, device=node_embeddings.device)
    per_graph_loss.index_add_(0, edge_batch, -torch.log(recall_final + 1e-8))
    
    # Normalize by edge count per graph
    counts = torch.bincount(edge_batch, minlength=num_graphs).float()
    counts[counts == 0] = 1  # avoid division by zero
    
    return (per_graph_loss / counts).mean()
