# Intramodel Search Strategy Benchmarking
Note that, the benchmarking program largely depends on macOS/XNU Specific File Handling, specifically Mach-O File Format. Change the file handling components to match the target system, in order to run these benchmarks 


### Why Bayesian Optimization Is Typically Slower Than Random Search
Even in real-world applications, Bayesian Optimization tends to be significantly slower than Random Search. This is due to several inherent computational and structural characteristics, as commonly noted in online resources and from ChatGPT:

- Surrogate Model Construction: Bayesian Optimization builds and updates a probabilistic surrogate model—most commonly a Gaussian Process—after each function evaluation. As the number of samples grows, the cost of fitting this model increases substantially.
- Acquisition Function Optimization: After updating the surrogate, the algorithm must optimize an acquisition function to select the next query point. This step introduces an additional layer of optimization, which becomes more demanding in high-dimensional spaces.
- Sequential Nature: Bayesian Optimization operates in a largely sequential manner, where each new evaluation depends on prior results. This dependency limits the ability to parallelize the search, unlike Random Search, which can evaluate multiple configurations simultaneously.
- Computational Scaling: Gaussian Processes, a standard choice for the surrogate model, have a computational complexity of $O(n^3)$ with respect to the number of observations. As more data is collected, this quickly becomes a computational bottleneck.

According to research and practical experience in the field:

- For problems with low-dimensional search spaces (fewer than 5 hyperparameters), Bayesian Optimization can be reasonably efficient.
- For higher-dimensional problems, the overhead becomes substantial. This matches our benchmark where we're seeing 100-200× slower performance.
- Many organizations and practitioners opt for Random Search or hybrid approaches in production because of these time constraints.


Typically industries tend to be employing hybrid approaches like **Tree-structured Parzen Estimators (TPE)** [https://arxiv.org/abs/2304.11127] that scale better than full Gaussian Processes
