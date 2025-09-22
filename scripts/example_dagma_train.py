"""
A prototype script to demonstrate the usage of the DAGMAScheduler callback
with a simple PyTorch Lightning training pipeline.
"""
# =============================================================================
# LOCAL IMPORTS
# =============================================================================
from library.losses.crl_loss import CausalRepresentationLearningLoss, CRLForwardPassOutput
from library.callbacks.dagma_scheduler import DAGMAScheduler

# =============================================================================
# TRAINING SCRIPT
# =============================================================================

class SimpleModel(L.LightningModule):
    """A dummy LightningModule to demonstrate the scheduler usage."""
    def __init__(self, latent_dim: int):
        super().__init__()
        # This parameter represents the graph matrix we want to learn
        self.G = torch.nn.Parameter(torch.zeros(latent_dim, latent_dim))
        
        # The loss function's `dagma_mu` will be controlled by the scheduler.
        self.loss_fn = CausalRepresentationLearningLoss(
            latent_dim=latent_dim,
            is_variational=False,
            is_causal=True,
            graph_loss_type="dagma",
            ret_dict=True,
        )
        # By calling `self.save_hyperparameters()`, we make `latent_dim` accessible
        # later, for example, in callbacks or logging.
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # In a real scenario, `batch` would come from the DataLoader.
        # Here, `x` is the input and `z_obs` is the latent representation
        # that DAGMA uses to score the graph `G`.
        x, z_obs = batch
        
        # We simulate the model's forward pass output. In a real model,
        # an encoder would produce `z_obs` from `x`.
        outputs = CRLForwardPassOutput(
            x_recon=x.clone(), # Dummy reconstruction
            y_hat=None, 
            mu=None, 
            log_var=None,
            G=self.G, 
            z_obs=z_obs, 
            z=None
        )

        # The loss function calculates all relevant components
        loss_dict = self.loss_fn.compute_loss(outputs, x, y=None)
        
        # Combine the losses for the backward pass
        total_loss = loss_dict.get("recon_loss", 0) + loss_dict.get("graph_loss", 0)
        
        # Log the metrics, which the scheduler will monitor
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=False)
        return total_loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

def main():
    """Runs the complete training and demonstration process."""
    print("--- Setting up the Trainer with the DAGMAScheduler ---")
    
    latent_dim = 10
    batch_size = 64
    
    # --- 1. Create the Model ---
    model = SimpleModel(latent_dim=latent_dim)
    
    # --- 2. Create Dummy Data ---
    # DAGMA's score function needs a dataset `X` (here `z_obs`) to evaluate the graph.
    # The dimensions are (num_samples, latent_dim).
    dummy_inputs = torch.randn(batch_size * 10, 100) # Dummy high-dimensional input
    dummy_latents = torch.randn(batch_size * 10, latent_dim) # Dummy latent representation
    dummy_dataset = TensorDataset(dummy_inputs, dummy_latents)
    train_loader = DataLoader(dummy_dataset, batch_size=batch_size)
    
    # --- 3. Instantiate the Scheduler Callback ---
    # This callback will watch the 'dagma_h_value' metric and update
    # the 'dagma_mu' parameter inside `model.loss_fn` accordingly.
    dagma_scheduler = DAGMAScheduler(
        mu_init=1.0,
        mu_update_factor=2.0, # Using a smaller factor for smoother increase
        h_threshold=1e-3,
        mu_max=1e+16,
    )
    
    # --- 4. Create the Trainer ---
    # The scheduler is passed in the `callbacks` list.
    trainer = L.Trainer(
        callbacks=[dagma_scheduler],
        max_epochs=5,
        log_every_n_steps=1,
    )
    
    # --- 5. Start Training ---
    print("\nStarting training...")
    trainer.fit(model, train_dataloaders=train_loader)
    print("\n--- Training complete ---")
    print(f"Final 'dagma_mu' value in loss function: {model.loss_fn.dagma_mu}")

if __name__ == '__main__':
    main()
