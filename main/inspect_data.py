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
        
        # Try different y file naming conventions
        y_file_variants = [
            x_file.replace('X_', 'Y_'),  # X_ -> Y_
            x_file.replace('X_', 'y_'),  # X_ -> y_
            'y' + x_file[1:],            # X... -> y...
            x_file.replace('X', 'y', 1)   # Replace first X with y
        ]
        
        y_path = None
        for y_file in y_file_variants:
            test_path = os.path.join(y_dir, y_file)
            if os.path.exists(test_path):
                y_path = test_path
                break
                
        if y_path is None:
            print(f"Error: Could not find corresponding y file for {x_file}")
            print(f"Tried variants: {', '.join(y_file_variants)}")
            return False
            
        try:
            # Load tensors
            x = torch.load(os.path.join(x_dir, x_file))
            y = torch.load(y_path)
            
            # Ensure 2D tensors
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)
            
            # Check each sequence in the batch
            all_aligned = True
            for i in range(min(x.size(0), num_samples)):
                x_seq = x[i]
                y_seq = y[i]
                
                # Check if y is x shifted left by 1
                x_shifted = x_seq[1:].cpu()
                y_expected = y_seq[:-1].cpu()
                
                # Check if all elements match
                if torch.all(x_shifted == y_expected):
                    print(f"Sample {i+1}: ✓ (Perfect alignment! Y is X shifted left by 1)")
                else:
                    all_aligned = False
                    print(f"\nSample {i+1}: ✗ Mismatch found!")
                    
                    # Find first mismatch
                    for j in range(min(len(x_shifted), len(y_expected))):
                        if x_shifted[j] != y_expected[j]:
                            start = max(0, j-2)
                            end = min(len(x_seq), j+3)
                            print("First mismatch at position:", j)
                            print("Context (position: X -> Y):")
                            for k in range(start, end):
                                x_val = x_seq[k].item()
                                y_val = y_seq[k].item() if k < len(y_seq) else "N/A"
                                mark = "  " if k != j+1 else "!!"  # Mark the position of the mismatch
                                print(f"{mark} pos {k:3d}: {x_val:5d} -> {y_val:5d}")
                            break
                    
                    # Show first 10 tokens for reference
                    print("\nFirst 10 tokens for reference:")
                    print(f"X: {x_seq[:10].tolist()}")
                    print(f"Y: {y_seq[:10].tolist()}")
                    print(f"Expected Y: {x_seq[1:11].tolist()}")
                    
                    # Show where they differ
                    print("\nPositions where X[1:] != Y[:-1]:")
                    for k in range(min(10, len(x_shifted))):
                        if x_shifted[k] != y_expected[k]:
                            print(f"  pos {k}: X[{k+1}]={x_shifted[k].item()} != Y[{k}]={y_expected[k].item()}")
            
            if all_aligned:
                print("\n✅ All samples verified successfully! The data is properly aligned for next-token prediction.")
            else:
                print("\n❌ Some samples are not properly aligned.")
            
            return all_aligned
            
        except Exception as e:
            print(f"Error processing files: {str(e)}")
            import traceback
            traceback.print_exc()
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