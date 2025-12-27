from math import exp, log

# ============================================
# HELPER FUNCTIONS (Matrix Operations)
# ============================================

def matmul(A, B):
    """Matrix multiplication A @ B (not using NumPy)"""
    m, n, p = len(A), len(A[0]), len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(A):
    """Matrix transpose"""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def add_bias(X, b):
    """Add bias vector b to each row of X"""
    return [[X[i][j] + b[j] for j in range(len(b))] for i in range(len(X))]

def sum_rows(X):
    """Sum along axis 0 (columns)"""
    return [sum(X[i][j] for i in range(len(X))) for j in range(len(X[0]))]

# ============================================
# ACTIVATION & LOSS FUNCTIONS
# ============================================

def relu(z):
    return max(0.0, z)

def relu_derivative(z):
    return 1.0 if z > 0.0 else 0.0

def sigmoid(z):
    if z < -100:  # Numerical stability
        return 0.0
    return 1.0 / (1.0 + exp(-z))

# ============================================
# TRAINING STEP (FORWARD + BACKWARD)
# ============================================

def training_step(X_batch, y_batch, W1, b1, W2, b2, learning_rate=0.01):
    """
    One complete training step for binary classifier
    batch_size = 2, input_dim = 3, hidden_dim = 4
    """
    batch_size = len(X_batch)
    
    # ========================================
    # FORWARD PASS
    # ========================================
    
    # 1. Hidden layer: z1 = X @ W1 + b1
    z1 = matmul(X_batch, W1)          # Shape: 2×4
    z1 = add_bias(z1, b1)             # Shape: 2×4
    
    # 2. ReLU activation: a1 = ReLU(z1)
    a1 = [[relu(z) for z in row] for row in z1]  # Shape: 2×4
    
    # 3. Output layer: z2 = a1 @ W2 + b2
    z2_raw = matmul(a1, W2)           # Shape: 2×1
    z2 = [row[0] + b2 for row in z2_raw]  # Shape: 2
    
    # 4. Sigmoid: y_hat = σ(z2)
    y_hat = [sigmoid(z) for z in z2]  # Shape: 2
    
    # 5. COMPUTE LOSS (BINARY CROSS-ENTROPY)
    # L = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
    epsilon = 1e-15
    total_loss = 0.0
    for i in range(batch_size):
        yh = max(epsilon, min(1-epsilon, y_hat[i]))  # Clip for stability
        total_loss -= y_batch[i] * log(yh) + (1-y_batch[i]) * log(1-yh)
    
    loss = total_loss / batch_size  # Average loss
    
    # ========================================
    # BACKWARD PASS
    # ========================================
    
    # Initialize gradients
    dW1 = [[0.0 for _ in range(len(W1[0]))] for _ in range(len(W1))]
    db1 = [0.0 for _ in range(len(b1))]
    dW2 = [[0.0 for _ in range(len(W2[0]))] for _ in range(len(W2))]
    db2 = 0.0
    
    # Process each sample
    for i in range(batch_size):
        # Gradient w.r.t z2: ∂L/∂z2 = y_hat - y  (THE KEY SIMPLIFICATION!)
        dL_dz2 = y_hat[i] - y_batch[i]  # This is the magic from Q2.2
        
        # Gradient for b2
        db2 += dL_dz2
        
        # Gradient for W2: ∂L/∂W2 = a1^T @ ∂L/∂z2
        for j in range(len(W2)):
            dW2[j][0] += a1[i][j] * dL_dz2
        
        # Gradient w.r.t a1: ∂L/∂a1 = ∂L/∂z2 @ W2^T
        dL_da1 = [dL_dz2 * W2[j][0] for j in range(len(W2))]
        
        # Gradient w.r.t z1: ∂L/∂z1 = ∂L/∂a1 ⊙ ReLU'(z1)
        dL_dz1 = [dL_da1[j] * relu_derivative(z1[i][j]) 
                  for j in range(len(z1[i]))]
        
        # Gradient for b1
        for j in range(len(db1)):
            db1[j] += dL_dz1[j]
        
        # Gradient for W1: ∂L/∂W1 = X^T @ ∂L/∂z1
        for j in range(len(W1)):
            for k in range(len(W1[0])):
                dW1[j][k] += X_batch[i][j] * dL_dz1[k]
    
    # Average gradients over batch
    dW1 = [[dW1[j][k] / batch_size for k in range(len(dW1[0]))] 
           for j in range(len(dW1))]
    db1 = [db1[j] / batch_size for j in range(len(db1))]
    dW2 = [[dW2[j][0] / batch_size] for j in range(len(dW2))]
    db2 /= batch_size
    
    # ========================================
    # GRADIENT UPDATE
    # ========================================
    
    # Update W1: W1 = W1 - lr * ∂L/∂W1
    for j in range(len(W1)):
        for k in range(len(W1[0])):
            W1[j][k] -= learning_rate * dW1[j][k]
    
    # Update b1: b1 = b1 - lr * ∂L/∂b1
    for j in range(len(b1)):
        b1[j] -= learning_rate * db1[j]
    
    # Update W2: W2 = W2 - lr * ∂L/∂W2
    for j in range(len(W2)):
        W2[j][0] -= learning_rate * dW2[j][0]
    
    # Update b2: b2 = b2 - lr * ∂L/∂b2
    b2 -= learning_rate * db2
    
    return loss, W1, b1, W2, b2