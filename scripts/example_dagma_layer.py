"""
PyTorch implementation of the DAGMA algorithm for learning Directed Acyclic Graphs (DAGs)
from observational data, structured with modular loss components.
"""
# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import numpy as np

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from library.losses.dagma_loss import DAGMALoss
from library.modules.dagma_linear import DAGMALinear, DAGMAPathFollowing

def set_seed(seed: int):
    """
    Sets the seed for all relevant random number generators to ensure
    reproducibility of results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def simulate_linear_sem(W, n, sem_type):
    d = W.shape[0]
    if sem_type == 'gauss':
        E = np.random.randn(n, d)
    else:
        raise ValueError("Unsupported sem_type")
    I_minus_W_T = np.eye(d) - W.T
    X = np.linalg.solve(I_minus_W_T, E.T).T
    return X

def simulate_dag(d, s0):
    B = np.tril(np.ones((d, d)), -1)
    p = np.random.permutation(np.arange(d))
    B_perm = B[p, :][:, p]
    
    non_zero_indices = np.where(B_perm != 0)
    edges_to_keep = np.random.choice(len(non_zero_indices[0]), s0, replace=False)
    B_final = np.zeros((d, d))
    B_final[non_zero_indices[0][edges_to_keep], non_zero_indices[1][edges_to_keep]] = 1
    return B_final

def simulate_parameter(B):
    return B * np.random.uniform(low=0.5, high=2.0, size=B.shape)

def calculate_shd(B_true, B_est):
    """Computes the Structural Hamming Distance."""
    diff = B_true - B_est
    #? Count additions and deletions
    add_del_count = np.sum(np.abs(diff))
    #? Count reversals
    reversal_mask = (diff.T == 1) & (diff == -1)
    reversal_count = np.sum(reversal_mask) // 2
    #? SHD is additions + deletions - reversals
    shd = add_del_count - reversal_count
    return shd

def test_small_scale_simulation():
    """
    A simple test function to demonstrate the usage of the DAGMALinear PyTorch module.
    """
    print("\n" + "#"*60)
    print("### RUNNING SMALL-SCALE SIMULATION (d=10) ###")
    print("#"*60)
    #? Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    #? --- Simulation ---
    n, d, s0 = 500, 10, 10
    B_true_np = simulate_dag(d, s0)
    W_true = simulate_parameter(B_true_np)
    X_np = simulate_linear_sem(W_true, n, 'gauss')
    #? Let PyTorch infer the dtype from the numpy array (float64 by default)
    X = torch.from_numpy(X_np)

    #? --- Model and Training ---
    model = DAGMALinear(d=d, verbose=True)
    dagma_loss = DAGMALoss(
        d=d,
        mu=0.1,
        lambda1=0.01
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Starting optimization...")
    model.train()
    for i in range(2000):
        optimizer.zero_grad()
        #? The model will handle casting X to its internal dtype if needed
        objective, score, h = dagma_loss(model.weight, X.to(model.weight.device, model.weight.dtype))
        if torch.isinf(objective) or torch.isnan(objective):
            break
        objective.backward()
        optimizer.step()
        model.enforce_zero_diagonal()
        if i % 400 == 0:
            model.vprint(f"Iter {i}: Objective={objective.item():.4f}, Score={score.item():.4f}, h(W)={h.item():.4f}")

    #? --- Results ---
    print("\n" + "="*50)
    print("RESULTS COMPARISON (d=10)")
    print("="*50)
    
    print(f"\nTrue Adjacency Matrix has {np.sum(B_true_np)} edges.")

    #? --- Config 1: Threshold, NO force_dag ---
    model.w_threshold = 0.3
    model.force_dag = False
    model.eval()
    B_est_1 = (model.W_thresholded.cpu().numpy() != 0).astype(int)
    shd_1 = calculate_shd(B_true_np, B_est_1)
    print(f"Config 1 (thresh=0.3, force=False): found {np.sum(B_est_1)} edges. SHD: {shd_1}")

    #? --- Config 2: Threshold AND force_dag ---
    model.w_threshold = 0.3
    model.force_dag = True
    model.eval()
    B_est_2 = (model.W_thresholded.cpu().numpy() != 0).astype(int)
    shd_2 = calculate_shd(B_true_np, B_est_2)
    print(f"Config 2 (thresh=0.3, force=True):  found {np.sum(B_est_2)} edges. SHD: {shd_2}")

    #? --- Config 3: NO threshold, automatic force_dag ---
    model.w_threshold = None
    model.force_dag = True #? This is automatically true
    model.eval()
    B_est_3 = (model.W_thresholded.cpu().numpy() != 0).astype(int)
    shd_3 = calculate_shd(B_true_np, B_est_3)
    print(f"Config 3 (thresh=None, force=True):  found {np.sum(B_est_3)} edges. SHD: {shd_3}")
    print("="*50)

def simulate_large_graph_with_subgraphs(num_nodes, num_subgraphs, s0_per_subgraph):
    """Creates a large block-diagonal adjacency matrix with multiple DAGs."""
    B_large = np.zeros((num_nodes, num_nodes))
    nodes_per_subgraph = num_nodes // num_subgraphs
    
    for i in range(num_subgraphs):
        start_idx = i * nodes_per_subgraph
        end_idx = (i + 1) * nodes_per_subgraph
        if i == num_subgraphs - 1: #? Ensure all nodes are used
            end_idx = num_nodes
        
        d_sub = end_idx - start_idx
        B_sub = simulate_dag(d_sub, s0_per_subgraph)
        B_large[start_idx:end_idx, start_idx:end_idx] = B_sub
        
    return B_large

def test_large_scale_simulation():
    """
    Test function for a large graph with disconnected DAG components.
    """
    print("\n" + "#"*60)
    print("### RUNNING LARGE-SCALE SIMULATION (d=1000) ###")
    print("#"*60)
    #? Set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    #? --- Simulation ---
    d = 1000
    num_subgraphs = 3
    s0_per_subgraph = 50
    n = 2000 #? More samples for a larger graph
    
    B_true_np = simulate_large_graph_with_subgraphs(d, num_subgraphs, s0_per_subgraph)
    W_true = simulate_parameter(B_true_np)
    X_np = simulate_linear_sem(W_true, n, 'gauss')
    X = torch.from_numpy(X_np)

    #? --- Model and Training ---
    model = DAGMALinear(d=d, verbose=True)
    dagma_loss = DAGMALoss(d=d, lambda1=0.02) #? Slightly lower L1 for sparser ground truth
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("Starting optimization for large graph...")
    start_time = time.time()
    model.train()
    #? Reduced iterations for demonstration purposes on a large graph
    for i in range(500):
        optimizer.zero_grad()
        objective, score, h = dagma_loss(model.weight, X.to(model.weight.device, model.weight.dtype))
        if torch.isinf(objective) or torch.isnan(objective):
            break
        objective.backward()
        optimizer.step()
        model.enforce_zero_diagonal()
        if i % 100 == 0:
            model.vprint(f"Iter {i}: Objective={objective.item():.4f}, Score={score.item():.4f}, h(W)={h.item():.4f}")
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    #? --- Results ---
    print("\n" + "="*50)
    print("RESULTS COMPARISON (d=1000)")
    print("="*50)
    
    true_edge_count = np.sum(B_true_np)
    print(f"\nTrue Adjacency Matrix has {true_edge_count} edges.")

    #? --- Config 1: Threshold, NO force_dag ---
    model.w_threshold = 0.3
    model.force_dag = False
    model.eval()
    B_est_1 = (model.W_thresholded.cpu().numpy() != 0).astype(int)
    shd_1 = calculate_shd(B_true_np, B_est_1)
    print(f"Config 1 (thresh=0.3, force=False): found {np.sum(B_est_1)} edges. SHD: {shd_1}")

    #? --- Config 2: Threshold AND force_dag ---
    model.w_threshold = 0.3
    model.force_dag = True
    model.eval()
    B_est_2 = (model.W_thresholded.cpu().numpy() != 0).astype(int)
    shd_2 = calculate_shd(B_true_np, B_est_2)
    print(f"Config 2 (thresh=0.3, force=True):  found {np.sum(B_est_2)} edges. SHD: {shd_2}")

    #? --- Config 3: NO threshold, automatic force_dag ---
    model.w_threshold = None
    model.force_dag = True #? This is automatically true
    model.eval()
    B_est_3 = (model.W_thresholded.cpu().numpy() != 0).astype(int)
    shd_3 = calculate_shd(B_true_np, B_est_3)
    print(f"Config 3 (thresh=None, force=True):  found {np.sum(B_est_3)} edges. SHD: {shd_3}")
    print("="*50)

def test_forward_pass_simulation():
    """
    Tests the forward() method by generating data from a known graph and noise.
    """
    print("\n" + "#"*60)
    print("### RUNNING FORWARD PASS SIMULATION (d=20) ###")
    print("#"*60)
    set_seed(42)
    
    #? --- 1. Setup ground truth ---
    n, d, s0 = 100, 20, 15
    B_true_np = simulate_dag(d, s0)
    W_true_np = simulate_parameter(B_true_np)
    W_true = torch.from_numpy(W_true_np).float()
    E = torch.randn(n, d).float()

    #? --- 2. Instantiate model and set its state ---
    model = DAGMALinear(d=d, w_threshold=None, force_dag=False)
    # Manually set the model's learned weight to the ground truth weight
    with torch.no_grad():
        model.weight.copy_(W_true)
    
    # Use eval mode to ensure W_thresholded is set and used
    model.eval()

    #? --- 3. Generate data using the model's forward pass ---
    X_pred = model(E)
    
    #? --- 4. Manually generate data for verification ---
    # We solve X(I - W^T) = E  => (I - W)^T X^T = E^T
    Id = torch.eye(d, dtype=W_true.dtype)
    I_minus_W_T = Id - W_true.T
    X_true = torch.linalg.solve(I_minus_W_T, E.T).T
    
    #? --- 5. Compare results ---
    print("Verifying model's forward pass against manual calculation...")
    reconstruction_error = torch.mean((X_pred - X_true)**2).item()
    print(f"Mean Squared Error between model output and ground truth: {reconstruction_error:.10f}")
    
    # Assert that the outputs are very close
    assert torch.allclose(X_pred, X_true, atol=1e-5), "Forward pass output does not match ground truth!"
    print("SUCCESS: The forward() method correctly solves the SEM.")
    print("#"*60)

def test_dagma_path_following():
    print("\n" + "#"*60)
    print("### RUNNING DAGMA PATH-FOLLOWING SIMULATION (d=10) ###")
    print("#"*60)
    torch.manual_seed(0); np.random.seed(0)

    n, d, s0 = 500, 10, 10
    B_true_np = simulate_dag(d, s0)
    W_true = simulate_parameter(B_true_np)
    X = torch.from_numpy(simulate_linear_sem(W_true, n, 'gauss'))

    model = DAGMAPathFollowing(d=d, verbose=True, lambda1=0.02)
    start_time = time.time()
    # The fit method now returns the raw, unthresholded weight matrix
    W_est_raw = model.fit(X, mu_steps=4, inner_steps=500)
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

    print("\n" + "="*50)
    print("RESULTS COMPARISON (d=10)")
    print("="*50)
    print(f"\nTrue Adjacency Matrix has {np.sum(B_true_np)} edges.")

    #? --- Config 1: Threshold, NO force_dag ---
    model.linear_model.w_threshold = 0.3
    model.linear_model.force_dag = False
    model.linear_model.eval()
    B_est_1 = (model.get_adjacency().cpu().numpy() != 0).astype(int)
    shd_1 = calculate_shd(B_true_np, B_est_1)
    print(f"Config 1 (thresh=0.3, force=False): found {np.sum(B_est_1)} edges. SHD: {shd_1}")

    #? --- Config 2: Threshold AND force_dag ---
    model.linear_model.w_threshold = 0.3
    model.linear_model.force_dag = True
    model.linear_model.eval()
    B_est_2 = (model.get_adjacency().cpu().numpy() != 0).astype(int)
    shd_2 = calculate_shd(B_true_np, B_est_2)
    print(f"Config 2 (thresh=0.3, force=True):  found {np.sum(B_est_2)} edges. SHD: {shd_2}")

    #? --- Config 3: NO threshold, automatic force_dag ---
    model.linear_model.w_threshold = None
    model.linear_model.force_dag = True # This is automatically true
    model.linear_model.eval()
    B_est_3 = (model.get_adjacency().cpu().numpy() != 0).astype(int)
    shd_3 = calculate_shd(B_true_np, B_est_3)
    print(f"Config 3 (thresh=None, force=True):  found {np.sum(B_est_3)} edges. SHD: {shd_3}")
    print("="*50)

if __name__ == '__main__':
    test_forward_pass_simulation()
    test_dagma_path_following()
    test_small_scale_simulation()
    # test_large_scale_simulation()