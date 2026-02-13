# Graph Neural Networks - Complete Guide (2025)

## Overview

**Graph Neural Networks (GNNs)** enable deep learning on graph-structured data - social networks, molecules, knowledge graphs, traffic networks, and more.

**Key Architectures:**
- Graph Convolutional Networks (GCN) - Foundational
- Graph Attention Networks (GAT) - Attention-based
- GraphSAGE - Scalable sampling
- Graph Transformers - State-of-the-art 2025

**Applications:** Molecular design, recommendation systems, traffic prediction, social network analysis, knowledge graphs

---

## Graph Basics

### Graph Representation

```python
import networkx as nx
import numpy as np

# Create graph
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])

# Adjacency matrix
A = nx.adjacency_matrix(G).todense()
print("Adjacency Matrix:\n", A)

# Node features (one-hot encoding for this example)
X = np.eye(len(G.nodes()))  # 5x5 identity matrix
print("Node Features:\n", X)

# Degree matrix
D = np.diag([G.degree(n) for n in G.nodes()])
print("Degree Matrix:\n", D)
```

---

## Graph Convolutional Networks (GCN)

### Theory

**GCN Layer:** Aggregate neighbor information using graph structure

**Formula:** H^(l+1) = sigma(D_tilde^(-1/2) A_tilde D_tilde^(-1/2) H^(l) W^(l))

Where:
- A_tilde = A + I (adjacency + self-loops)
- D_tilde = degree matrix of A_tilde
- H^(l) = node features at layer l
- W^(l) = learnable weights
- sigma = activation function

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        """
        Args:
            X: Node features (N x in_features)
            A: Normalized adjacency matrix (N x N)
        """
        # Aggregate neighbor features
        aggregated = torch.mm(A, X)  # A @ X

        # Transform
        output = self.linear(aggregated)

        return output

class GCN(nn.Module):
    """
    Full GCN model
    """
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, X, A):
        # Layer 1
        h = self.gcn1(X, A)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        # Layer 2
        h = self.gcn2(h, A)

        return F.log_softmax(h, dim=1)

# Preprocessing: Normalize adjacency matrix
def normalize_adjacency(A):
    """
    Compute D^(-1/2) @ A @ D^(-1/2)
    """
    # Add self-loops
    A_hat = A + torch.eye(A.size(0))

    # Degree matrix
    D = torch.diag(A_hat.sum(1))

    # D^(-1/2)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag()))

    # Normalized adjacency
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    return A_norm

# Usage
A_norm = normalize_adjacency(torch.FloatTensor(A))
X_tensor = torch.FloatTensor(X)

model = GCN(num_features=5, hidden_dim=16, num_classes=3)
output = model(X_tensor, A_norm)
```

---

## Graph Attention Networks (GAT)

### Attention Mechanism for Graphs

**Key Idea:** Learn importance of neighbors (not all neighbors equally important)

**Attention Coefficient:** alpha_ij = attention from node i to j

```python
class GATLayer(nn.Module):
    """
    Graph Attention Layer
    """
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        # Multi-head attention
        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, X, A):
        """
        Args:
            X: Node features (N x in_features)
            A: Adjacency matrix (N x N)
        """
        N = X.size(0)

        # Linear transformation
        h = torch.mm(X, self.W).view(N, self.num_heads, self.out_features)  # (N, heads, out)

        # Attention mechanism
        a_input = torch.cat([
            h.repeat(1, 1, N).view(N * N, self.num_heads, self.out_features),
            h.repeat(N, 1, 1)
        ], dim=2).view(N, N, self.num_heads, 2 * self.out_features)

        # Attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (N, N, heads)

        # Mask attention by adjacency (only attend to neighbors)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A.unsqueeze(2) > 0, e, zero_vec)

        # Softmax
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Aggregate with attention
        h_prime = torch.matmul(attention.transpose(1, 2), h)  # (N, heads, out)

        # Concatenate or average heads
        h_prime = h_prime.view(N, self.num_heads * self.out_features)

        return F.elu(h_prime)

class GAT(nn.Module):
    """
    Full GAT model
    """
    def __init__(self, num_features, hidden_dim, num_classes, num_heads=8):
        super().__init__()
        self.gat1 = GATLayer(num_features, hidden_dim, num_heads=num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, num_classes, num_heads=1)

    def forward(self, X, A):
        h = self.gat1(X, A)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, A)
        return F.log_softmax(h, dim=1)

# Usage
model = GAT(num_features=5, hidden_dim=8, num_classes=3, num_heads=8)
output = model(X_tensor, torch.FloatTensor(A))
```

**Advantages over GCN:**
- Learns which neighbors are important
- Different attention for different neighbors
- Better performance on many tasks

---

## GraphSAGE - Scalable GNN

### Sampling and Aggregating

**Problem:** GCN/GAT require full graph in memory (doesn't scale to billions of nodes)

**Solution:** Sample fixed-size neighborhood

```python
class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer with neighborhood sampling
    """
    def __init__(self, in_features, out_features, aggregator='mean'):
        super().__init__()
        self.aggregator = aggregator

        # Aggregator-specific transformations
        if aggregator == 'mean':
            self.agg_linear = nn.Linear(in_features, out_features)
            self.self_linear = nn.Linear(in_features, out_features)
        elif aggregator == 'pool':
            self.pool_linear = nn.Linear(in_features, in_features)
            self.agg_linear = nn.Linear(in_features, out_features)
            self.self_linear = nn.Linear(in_features, out_features)

    def forward(self, X, neighbors_dict):
        """
        Args:
            X: Node features
            neighbors_dict: {node_id: [neighbor_ids]} (sampled)
        """
        N = X.size(0)
        aggregated = torch.zeros(N, X.size(1))

        for node_id in range(N):
            neighbor_ids = neighbors_dict.get(node_id, [])

            if len(neighbor_ids) == 0:
                aggregated[node_id] = X[node_id]
                continue

            # Get neighbor features
            neighbor_features = X[neighbor_ids]

            # Aggregate
            if self.aggregator == 'mean':
                aggregated[node_id] = neighbor_features.mean(0)
            elif self.aggregator == 'pool':
                # Max pooling after linear transformation
                pooled = F.relu(self.pool_linear(neighbor_features))
                aggregated[node_id] = pooled.max(0)[0]

        # Combine self and aggregated
        if self.aggregator == 'mean':
            output = F.relu(
                self.self_linear(X) + self.agg_linear(aggregated)
            )
        elif self.aggregator == 'pool':
            output = F.relu(
                self.self_linear(X) + self.agg_linear(aggregated)
            )

        # Normalize
        output = F.normalize(output, p=2, dim=1)

        return output

def sample_neighbors(G, nodes, num_samples=10):
    """
    Sample fixed number of neighbors for each node
    """
    neighbors_dict = {}

    for node in nodes:
        neighbors = list(G.neighbors(node))

        # Sample
        if len(neighbors) > num_samples:
            sampled = np.random.choice(neighbors, num_samples, replace=False)
        else:
            sampled = neighbors

        neighbors_dict[node] = sampled

    return neighbors_dict

# Usage - Scalable to billions of nodes!
model = GraphSAGELayer(in_features=128, out_features=64, aggregator='mean')

# Sample neighbors (fixed size)
neighbors = sample_neighbors(G, nodes=range(len(G.nodes())), num_samples=10)

# Forward pass (only uses sampled neighbors)
output = model(X_tensor, neighbors)
```

**Performance:** Outperforms GAT and GCN, even on small datasets. Scales to 3B nodes (Pinterest PinSAGE)

---

## Applications

### 1. Molecular Property Prediction

```python
from rdkit import Chem
from rdkit.Chem import rdmolops

def mol_to_graph(smiles):
    """
    Convert molecule (SMILES) to graph
    """
    mol = Chem.MolFromSmiles(smiles)

    # Nodes = atoms
    num_atoms = mol.GetNumAtoms()

    # Node features (atom type, charge, etc.)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetTotalDegree(),  # Degree
            atom.GetFormalCharge(),  # Charge
            atom.GetHybridization(),  # Hybridization
            atom.GetIsAromatic()  # Aromatic
        ]
        node_features.append(features)

    X = torch.FloatTensor(node_features)

    # Edges = bonds
    adjacency = rdmolops.GetAdjacencyMatrix(mol)
    A = torch.FloatTensor(adjacency)

    return X, A

# Predict molecular property (e.g., toxicity, solubility)
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
X, A = mol_to_graph(smiles)

# GNN for molecular property prediction
class MolecularGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(5, 64)  # 5 atom features
        self.gcn2 = GCNLayer(64, 32)
        self.fc = nn.Linear(32, 1)  # Predict single property

    def forward(self, X, A):
        h = F.relu(self.gcn1(X, normalize_adjacency(A)))
        h = F.relu(self.gcn2(h, normalize_adjacency(A)))

        # Graph-level prediction (pool node features)
        h_graph = h.mean(0)  # Global mean pooling

        return self.fc(h_graph)

mol_model = MolecularGNN()
toxicity = mol_model(X, A)
print(f"Predicted toxicity: {toxicity.item():.3f}")
```

**Applications:**
- Drug discovery
- Material science
- Chemical reaction prediction

---

### 2. Recommendation Systems (PinSAGE)

```python
class PinSAGE(nn.Module):
    """
    Pinterest's recommendation GNN
    Handles 3B nodes, 18B edges
    """
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.sage1 = GraphSAGELayer(num_features, hidden_dim, 'mean')
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim, 'mean')

    def forward(self, X, neighbors_dict_layer1, neighbors_dict_layer2):
        # Layer 1: Sample and aggregate
        h = self.sage1(X, neighbors_dict_layer1)

        # Layer 2: Sample and aggregate from layer 1 embeddings
        h = self.sage2(h, neighbors_dict_layer2)

        return h

# Graph: Users and Items as nodes, Interactions as edges
# Node features: User demographics, Item attributes
# Predict: Which items to recommend

pin_sage = PinSAGE(num_features=256, hidden_dim=128)

# Get embeddings
item_embeddings = pin_sage(item_features, neighbors_layer1, neighbors_layer2)
user_embeddings = pin_sage(user_features, neighbors_layer1, neighbors_layer2)

# Recommendation score = dot product
scores = torch.mm(user_embeddings, item_embeddings.t())
top_k_items = scores.topk(k=10, dim=1)
```

---

### 3. Traffic Prediction (Spatial-Temporal GNN)

```python
class STGNN(nn.Module):
    """
    Spatial-Temporal Graph Neural Network
    For traffic speed/volume prediction
    """
    def __init__(self, num_nodes, num_features, hidden_dim):
        super().__init__()
        # Spatial: GCN
        self.gcn = GCNLayer(num_features, hidden_dim)

        # Temporal: LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output
        self.fc = nn.Linear(hidden_dim, 1)  # Predict speed

    def forward(self, X_seq, A):
        """
        Args:
            X_seq: (batch, time_steps, num_nodes, features)
            A: Adjacency matrix (num_nodes, num_nodes)
        """
        batch_size, time_steps, num_nodes, num_features = X_seq.size()

        # Process each time step with GCN
        spatial_features = []
        for t in range(time_steps):
            h = self.gcn(X_seq[:, t], A)  # (batch, num_nodes, hidden)
            spatial_features.append(h)

        spatial_features = torch.stack(spatial_features, dim=1)  # (batch, time, nodes, hidden)

        # LSTM for temporal dependencies
        lstm_out, _ = self.lstm(spatial_features.view(batch_size * num_nodes, time_steps, -1))

        # Predict next time step
        predictions = self.fc(lstm_out[:, -1, :])  # (batch * nodes, 1)

        return predictions.view(batch_size, num_nodes)

# Usage: Predict traffic speed in 15 minutes
st_gnn = STGNN(num_nodes=100, num_features=5, hidden_dim=64)

# X_seq: Historical traffic data (last hour, 4 time steps)
# A: Road network adjacency
predicted_speed = st_gnn(X_seq, A)
```

---

## Graph Transformers (2025 SOTA)

### Attention Over Graph Structure

```python
class GraphTransformerLayer(nn.Module):
    """
    Transformer for graphs with positional encoding
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        # Positional encoding from graph structure
        pos_encoding = self.laplacian_positional_encoding(A, d_model=X.size(1))
        X = X + pos_encoding

        # Self-attention
        attn_out, _ = self.attention(X, X, X)
        X = self.norm1(X + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(X)
        X = self.norm2(X + self.dropout(ffn_out))

        return X

    def laplacian_positional_encoding(self, A, d_model, k=10):
        """
        Use graph Laplacian eigenvectors as positional encoding
        """
        # Laplacian
        D = torch.diag(A.sum(1))
        L = D - A

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Use top k eigenvectors
        pos_enc = eigenvectors[:, :k]

        # Project to d_model dimensions
        if k < d_model:
            pos_enc = F.pad(pos_enc, (0, d_model - k))
        elif k > d_model:
            pos_enc = pos_enc[:, :d_model]

        return pos_enc
```

---

## Best Practices

1. **Choose architecture by task:**
   - GCN: Homophilic graphs (similar nodes connected)
   - GAT: Heterophilic graphs (different nodes connected)
   - GraphSAGE: Large graphs (billions of nodes)
   - Graph Transformer: Complex patterns, long-range dependencies

2. **Normalize adjacency** - Always add self-loops and normalize

3. **Handle large graphs** - Use sampling (GraphSAGE) or mini-batches

4. **Node vs Graph tasks:**
   - Node classification: Output per node
   - Graph classification: Pool node features --> single output

5. **Oversmoothing** - Too many GNN layers --> all nodes same embedding (use residual connections)

---

## Key Takeaways

- **GNNs enable deep learning on graphs** - Social networks, molecules, traffic, recommendations
- **GCN** - Foundational, simple aggregation
- **GAT** - Attention-based, learns neighbor importance
- **GraphSAGE** - Scalable to billions (sampling), best performance
- **Graph Transformers** - 2025 SOTA, long-range dependencies
- **Applications everywhere** - Drug discovery, recommendations, traffic, social analysis

**Next:** Time Series Deep Learning and AutoML/NAS guides
