import numpy as np
from typing import Union, Tuple, List
import matplotlib.pyplot as plt

class DTW:
    def __init__(self):
        """Initialize the DTW algorithm."""
        self.cost_matrix = None
        self.accumulated_cost_matrix = None
        self.path = None
    
    def calculate_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x - y) ** 2))
    
    def create_cost_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create the cost matrix between two sequences."""
        # Initialize the cost matrix
        n, m = np.shape(x)[0], np.shape(y)[0]
        cost_matrix = np.zeros((n, m))
        
        # Fill the cost matrix with distances
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = self.calculate_distance(x[i], y[j])
                
        return cost_matrix
    
    def compute_accumulated_cost_matrix(self, cost_matrix: np.ndarray) -> np.ndarray:
        """Compute the accumulated cost matrix."""
        n, m = cost_matrix.shape
        accumulated_cost = np.zeros((n, m))
        
        # Initialize first elements
        accumulated_cost[0, 0] = cost_matrix[0, 0]
        
        # Fill first row and column
        for i in range(1, n):
            accumulated_cost[i, 0] = accumulated_cost[i-1, 0] + cost_matrix[i, 0]
        for j in range(1, m):
            accumulated_cost[0, j] = accumulated_cost[0, j-1] + cost_matrix[0, j]
        
        # Fill the rest of the matrix
        for i in range(1, n):
            for j in range(1, m):
                accumulated_cost[i, j] = cost_matrix[i, j] + min(
                    accumulated_cost[i-1, j],    # insertion
                    accumulated_cost[i, j-1],    # deletion
                    accumulated_cost[i-1, j-1]   # match
                )
                
        return accumulated_cost
    
    def find_warping_path(self, accumulated_cost: np.ndarray) -> List[Tuple[int, int]]:
        """Find the optimal warping path through the accumulated cost matrix."""
        n, m = accumulated_cost.shape
        path = [(n-1, m-1)]
        
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((i, j-1))
            elif j == 0:
                path.append((i-1, j))
            else:
                possible_moves = [
                    (i-1, j-1),  # diagonal
                    (i-1, j),    # vertical
                    (i, j-1)     # horizontal
                ]
                costs = [accumulated_cost[move] for move in possible_moves]
                best_move = possible_moves[np.argmin(costs)]
                path.append(best_move)
        
        return path[::-1]  # Reverse the path to start from (0,0)
    
    def dtw(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Compute DTW distance between two sequences and return the warping path.
        
        Args:
            x: First sequence (n_samples x n_features)
            y: Second sequence (m_samples x n_features)
            
        Returns:
            distance: The DTW distance between sequences
            path: The optimal warping path
        """
        # Create cost matrix
        self.cost_matrix = self.create_cost_matrix(x, y)
        
        # Compute accumulated cost matrix
        self.accumulated_cost_matrix = self.compute_accumulated_cost_matrix(self.cost_matrix)
        
        # Find optimal warping path
        self.path = self.find_warping_path(self.accumulated_cost_matrix)
        
        # Return the DTW distance (final accumulated cost) and the path
        return self.accumulated_cost_matrix[-1, -1], self.path
    
    def plot_cost_matrix(self, title: str = "DTW Cost Matrix", show_path: bool = True):
        """Plot the cost matrix and optionally the warping path."""
        if self.accumulated_cost_matrix is None:
            raise ValueError("No DTW computation has been performed yet.")
            
        plt.figure(figsize=(10, 8))
        plt.imshow(self.accumulated_cost_matrix, origin='lower', cmap='viridis')
        plt.colorbar(label='Accumulated Cost')
        
        if show_path and self.path is not None:
            path_x, path_y = zip(*self.path)
            plt.plot(path_y, path_x, 'r-', linewidth=2, label='Optimal Path')
            plt.legend()
            
        plt.title(title)
        plt.xlabel('Sequence Y')
        plt.ylabel('Sequence X')
        plt.show()

class VowelIdentifier:
    def __init__(self):
        """Initialize the vowel identifier with DTW algorithm."""
        self.dtw = DTW()
        self.reference_features = {}
        
    def add_reference(self, vowel: str, features: np.ndarray):
        """Add reference features for a vowel."""
        self.reference_features[vowel] = features
        
    def identify(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Identify the vowel by comparing with reference features.
        
        Args:
            features: MFCC features of the input audio
            
        Returns:
            vowel: The identified vowel
            distance: The DTW distance to the best match
        """
        if not self.reference_features:
            raise ValueError("No reference features added yet.")
            
        best_vowel = None
        min_distance = float('inf')
        
        for vowel, ref_features in self.reference_features.items():
            distance, _ = self.dtw.dtw(features, ref_features)
            
            if distance < min_distance:
                min_distance = distance
                best_vowel = vowel
                
        return best_vowel, min_distance