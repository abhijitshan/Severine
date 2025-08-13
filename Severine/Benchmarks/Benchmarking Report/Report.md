//
//  Report.md
//  Severine
//
//  Created by Abhijit Shanbhag on 05/05/25.
//
# Comparison: Severine vs. Optuna - Linear Regression Benchmark


## Single-Thread Performance

**Severine (1 Thread):**
- Best Score: 0.9998303985
- Total Time: 2.58 seconds
- Peak Memory: 19.92 MB
- Best Config: {alpha=0.00011254, tol=0.00009077, fit_intercept=false, max_iter=4887, solver=lsqr}

**Optuna (1 Thread):**
- Best Score: 0.9999997
- Total Time: 8.41 seconds
- Peak Memory: 191.47 MB
- Best Config: {alpha=0.0020654, fit_intercept=True, max_iter=471, solver=lsqr, tol=0.00002413}

## Multi-Thread Performance

**Severine (10 Threads):**
- Best Score: 0.9998304003
- Total Time: 0.48 seconds
- Peak Memory: 27.30 MB
- Best Config: {alpha=0.00010474, tol=0.00002683, fit_intercept=false, max_iter=2539, solver=lsqr}

**Optuna (10 Threads):**
- Best Score: 0.9999997
- Total Time: 2.88 seconds
- Peak Memory: 196.78 MB
- Best Config: {alpha=0.0020495, fit_intercept=True, max_iter=2983, solver=lsqr, tol=0.0002629}

## Key Insights

1. **Score Comparison**:
   - Optuna still achieves marginally better R² scores (0.9999997 vs 0.9998304)
   - The gap has significantly narrowed with our LSQR implementation
   - Both achieve excellent fits (above 0.999)

2. **Time Efficiency**:
   - Severine is now significantly faster than Optuna in both single and multi-threaded modes
   - Single thread: Severine is 3.3× faster (2.58s vs 8.41s)
   - Multi-thread: Severine is 6× faster (0.48s vs 2.88s)

3. **Memory Usage**:
   - Severine remains dramatically more memory-efficient
   - Single thread: 19.92 MB vs 191.47 MB (9.6× less memory)
   - Multi-thread: 27.3 MB vs 196.78 MB (7.2× less memory)

4. **Scaling Efficiency**:
   - Severine shows excellent parallelization scaling: 5.4× speedup with 10 threads
   - Optuna shows good scaling: 2.9× speedup with 10 threads
   - Severine handles threads more efficiently

5. **Hyperparameter Selection**:
   - Both selected the LSQR solver
   - Similar tolerance values in the same order of magnitude
   - Different preferences for fit_intercept (Optuna: True, Severine: False)
   - Similar alpha values (light regularization)

## Conclusions

1. **Performance Improvement**: Our LSQR implementation dramatically improved Severine's performance, making it competitive with Optuna in terms of model quality.

2. **Efficiency Leader**: Severine is now definitively more efficient than Optuna in both time and memory usage. The efficiency gap becomes even more pronounced in multi-threaded scenarios.

3. **Memory Advantage**: Severine's memory efficiency (using ~1/9th of Optuna's memory) makes it much more suitable for resource-constrained environments.

4. **Parallelization**: Severine's superior thread scaling suggests a more efficient implementation for parallel hyperparameter tuning.

5. **Resource Tradeoff**: Optuna still achieves marginally better scores, but at a much higher computational cost that may not be justified for most practical applications.

The single most remarkable achievement is the 6× faster execution with 7× less memory while maintaining competitive model quality.
