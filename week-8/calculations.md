
# Week 8 – Attention Mechanisms

## Part 1: Scaled Dot-Product Attention

### 1. Input Matrices

Query Matrix (Q):

    Q = [[1, 0],
         [0, 1]]

Key Matrix (K):

    K = [[1, 0],
         [1, 1]]

Value Matrix (V):

    V = [[10, 0],
         [0, 20]]

### 2. Transpose of Key Matrix

    K transpose = [[1, 1],
                   [0, 1]]

### 3. Calculate Attention Scores

Formula:

    Scores = Q × K transpose

Calculation:

    [[1, 0],   [[1, 1],   [[1, 1],
     [0, 1]] ×  [0, 1]] =  [0, 1]]

### 4. Scale the Scores

Formula:

    Scaled Scores = Scores / sqrt(d_k)

Here, d_k = 2.

    sqrt(2) = 1.4142

Result:

    [[0.7071, 0.7071],
     [0.0000, 0.7071]]

### 5. Apply Softmax

Formula:

    Attention Weights = Softmax(Scaled Scores)

Result:

    [[0.5000, 0.5000],
     [0.3302, 0.6698]]

### 6. Calculate Final Attention Output

Formula:

    Output = Attention Weights × V

Result:

    [[5.0000, 10.0000],
     [3.3024, 13.3952]]

### 7. Final Result

The scaled dot-product attention calculation is completed successfully.

---

## Part 2: Multi-Head Attention

### 1. Input Matrix

    X = [[1, 0, 1, 1],
         [0, 1, 1, 0]]

Number of features = 4

Number of attention heads = 2

Features per head = 4 / 2 = 2

### 2. Split Input into Two Heads

Head 1 (first two columns):

    Q1 = K1 = V1 = [[1, 0],
                    [0, 1]]

Head 2 (last two columns):

    Q2 = K2 = V2 = [[1, 1],
                    [1, 0]]

In our example, the projection weights are identity matrices.
Therefore, Q = K = V = X.

### 3. Head 1 Calculations

Attention Scores:

    Q1 × K1 transpose

    [[1, 0],
     [0, 1]]

Scaled Scores (divide by sqrt(2)):

    [[0.7071, 0.0000],
     [0.0000, 0.7071]]

Softmax Weights:

    [[0.6698, 0.3302],
     [0.3302, 0.6698]]

Head 1 Output:

    [[0.6698, 0.3302],
     [0.3302, 0.6698]]

### 4. Head 2 Calculations

Attention Scores:

    Q2 × K2 transpose

    [[2, 1],
     [1, 1]]

Scaled Scores (divide by sqrt(2)):

    [[1.4142, 0.7071],
     [0.7071, 0.7071]]

Softmax Weights:

    [[0.6698, 0.3302],
     [0.5000, 0.5000]]

Head 2 Output:

    [[1.0000, 0.6698],
     [1.0000, 0.5000]]

### 5. Concatenate Both Head Outputs

Combine Head 1 and Head 2 outputs column-wise:

    [[0.6698, 0.3302, 1.0000, 0.6698],
     [0.3302, 0.6698, 1.0000, 0.5000]]

### 6. Final Linear Projection

The output projection weight matrix is an identity matrix.

Therefore, the final output remains:

    [[0.6698, 0.3302, 1.0000, 0.6698],
     [0.3302, 0.6698, 1.0000, 0.5000]]

### 7. Final Result

Multi-Head Attention is completed successfully.

Each attention head calculates attention separately.
Their outputs are concatenated to produce the final result.
