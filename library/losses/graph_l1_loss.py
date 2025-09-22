import torch

class GraphL1Loss:
    def __init__(self):
        """
        Initializes the L1LossFunction class.
        """
        pass
    
    def __call__(self, 
        G: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the L1 regularization loss for the adjacency matrix.

        Args:
            G (torch.Tensor): Adjacency matrix.

        Returns:
            torch.Tensor: L1 regularization loss.
        """
        if G is None:
            return torch.tensor(0.0)
            
        #? Calculate L1 regularization
        l1_norm = torch.norm(torch.triu(G, diagonal=1), p=1)
        mask = torch.triu(torch.ones_like(G), diagonal=1)
        total_elements = torch.sum(mask)
        
        return l1_norm / total_elements