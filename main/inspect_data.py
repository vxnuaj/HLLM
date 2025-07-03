import torch
import os

def verify_token_alignment(data_dir, num_samples=5):
    """
    Verify that for each sequence in X, the next token is correctly aligned in y.
    
    Args:
        data_dir: Directory containing 'X' and 'y' subdirectories with .pt files
        num_samples: Number of samples to check from each file
    """
    # Handle case where X and y are in separate subdirectories
    x_dir = os.path.join(data_dir, 'X')
    y_dir = os.path.join(data_dir, 'y')
    
    # Get all .pt files in X directory
    if not os.path.exists(x_dir):
        print(f"Error: X directory not found at {x_dir}")
        return False
        
    x_files = [f for f in os.listdir(x_dir) if f.endswith('.pt')]
    
    if not x_files:
        print(f"No .pt files found in {x_dir}")
        return False
    
    for x_file in x_files:
        print(f"\nChecking file: {x_file}")
        print("-" * 50)
        
        # Construct paths
        x_path = os.path.join(x_dir, x_file)
       
        y_file = x_file.replace('X_', 'Y_')  # Convert X_*.pt to Y_*.pt
            
            
        y_path = os.path.join(y_dir, y_file)
        
        if not os.path.exists(y_path):
            y_file = x_file.replace('X_', 'y_')            
            y_path = os.path.join(y_dir, y_file) 
            
            
        try:
            # Load tensors
            x = torch.load(x_path)
            y = torch.load(y_path)
            
            # Ensure 2D tensors
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)
            
            # Check each sequence in the batch
            for i in range(min(x.size(0), num_samples)):
                x_seq = x[i]
                y_seq = y[i]
                
                # Check if y is x shifted left by 1
                is_aligned = torch.all(x_seq[1:] == y_seq[:-1])
                
                print(f"Sample {i+1}: {'✓' if is_aligned else '✗'}")
                
                # Show first few tokens if not aligned
                if not is_aligned:
                    print("Mismatch found! First 10 tokens:")
                    print(f"X: {x_seq[:10].tolist()}")
                    print(f"Y: {y_seq[:10].tolist()}")
                    print("Expected Y:", x_seq[1:11].tolist())
                    return False
            
            print("All samples verified successfully!")
            return True
            
        except Exception as e:
            print(f"Error processing files: {str(e)}")
            return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inspect_data.py <data_directory> [num_samples]")
        print("Example: python inspect_data.py data/tensors/val 5")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    verify_token_alignment(data_dir, num_samples)