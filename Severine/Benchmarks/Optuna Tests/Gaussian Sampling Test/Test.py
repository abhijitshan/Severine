#!/usr/bin/env python3
"""
Optuna Benchmark Script

This script benchmarks Optuna's optimization strategies (Random, TPE, and GP)
on Linear Regression using synthetic datasets matching the HyperTune configuration.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import optuna
from optuna.samplers import RandomSampler, TPESampler

try:
    from optuna.samplers import GPSampler

    GP_AVAILABLE = True
except ImportError:
    print("GPSampler not available in your Optuna version. Will skip GP benchmarks.")
    GP_AVAILABLE = False

# Constants to match HyperTune settings
ITERATIONS = 100
THREADS = 10
OUTPUT_DIR = "optuna_benchmark_results"


# Class to track memory usage
class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.current_usage = 0
        self.peak_usage = 0

    def update(self):
        """Update memory usage statistics."""
        self.current_usage = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        self.peak_usage = max(self.peak_usage, self.current_usage)
        return self.current_usage


# Generate synthetic regression dataset (matching HyperTune)
def generate_synthetic_data(n_samples=1000, n_features=20, noise=0.1, random_state=42):
    """Generate synthetic regression data with specified characteristics."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# Create objective function for Linear Regression with matching hyperparameters
def create_objective_linear_regression(X_train, X_test, y_train, y_test):
    """Create objective function for Optuna to optimize Linear Regression."""

    def objective(trial):
        # Match HyperTune's hyperparameter space exactly
        alpha = trial.suggest_float("alpha", 0.0001, 1.0, log=True)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        max_iter = trial.suggest_int("max_iter", 100, 5000)
        tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
        solver = trial.suggest_categorical("solver", ["lsqr"])  # Force lsqr to match

        # Create and train model
        model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            solver=solver
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        return score  # Optuna maximizes by default

    return objective


# Benchmark result class
class BenchmarkResult:
    def __init__(self):
        self.scores = []
        self.best_score_history = []
        self.times = []
        self.memory_usage = []
        self.total_time = 0
        self.peak_memory = 0
        self.best_params = {}
        self.best_score = -float('inf')


# Run benchmark with specified sampler
def run_benchmark(
        sampler_name,
        X_train, X_test, y_train, y_test,
        n_trials=ITERATIONS,
        n_jobs=THREADS
):
    """Run Optuna benchmark with specified sampler."""

    print(f"\nRunning benchmark for {sampler_name} search...")

    # Create result object
    result = BenchmarkResult()

    # Setup memory monitoring
    memory_monitor = MemoryMonitor()

    # Create appropriate sampler
    if sampler_name == "random":
        sampler = RandomSampler(seed=42)
    elif sampler_name == "tpe":
        sampler = TPESampler(n_startup_trials=10, seed=42)
    elif sampler_name == "gp" and GP_AVAILABLE:
        sampler = GPSampler(n_startup_trials=10, seed=42)
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    # Create objective function
    objective = create_objective_linear_regression(X_train, X_test, y_train, y_test)

    # Custom callback to track progress
    current_best = -float('inf')
    start_time = time.time()

    def callback(study, trial):
        nonlocal current_best

        # Record score
        score = trial.value if trial.value is not None else float('-inf')
        result.scores.append(score)

        # Update best score
        current_best = max(current_best, score)
        result.best_score_history.append(current_best)

        # Record elapsed time
        elapsed = time.time() - start_time
        result.times.append(elapsed)

        # Monitor memory
        mem_usage = memory_monitor.update()
        result.memory_usage.append(mem_usage)

    # Create and run study
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize"
    )

    # Measure total time
    start = time.time()

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[callback]
    )

    # Record final results
    end = time.time()
    result.total_time = end - start
    result.peak_memory = memory_monitor.peak_usage
    result.best_params = study.best_params
    result.best_score = study.best_value

    print(f"... {sampler_name} search complete.")

    return result, study


# Save benchmark results to CSV in same format as HyperTune
def save_benchmark_to_csv(filename, result, sampler_name):
    """Save benchmark results to CSV files."""

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write main CSV file with scores and memory usage
    with open(filename, 'w') as f:
        # Write header
        f.write(f"Iteration,{sampler_name}_score,{sampler_name}_best_score,{sampler_name}_memory_mb\n")

        # Write data for each iteration
        for i in range(len(result.scores)):
            f.write(f"{i + 1},{result.scores[i]},{result.best_score_history[i]},{result.memory_usage[i]}\n")

    # Write summary to a separate file
    summary_filename = filename.replace('.csv', '_summary.csv')

    with open(summary_filename, 'w') as f:
        # Write header
        f.write("Strategy,Best Score,Total Time (s),Peak Memory (MB),Best Config\n")

        # Format best config as a string
        config_str = "; ".join([f"{key}={value}" for key, value in result.best_params.items()])

        # Write data
        f.write(f"{sampler_name},{result.best_score},{result.total_time},{result.peak_memory},\"{config_str}\"\n")

    return filename, summary_filename


# Print benchmark summary in same format as HyperTune
def print_benchmark_summary(result, sampler_name):
    """Print a summary of benchmark results."""

    print("\n=== Benchmark Summary ===\n")
    print(f"{'Strategy':<15}{'Best Score':<20}{'Total Time (s)':<16}{'Peak Memory (MB)':<20}Best Configuration")
    print("-" * 100)

    # Format best config as a string
    config_str = "{ " + ", ".join([f"{key}={value}" for key, value in result.best_params.items()]) + " }"

    print(
        f"{sampler_name:<15}{result.best_score:<20.10f}{result.total_time:<16.4f}{result.peak_memory:<20.2f}{config_str}")
    print("-" * 100)


# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optuna Benchmarking Tool")

    parser.add_argument("--sampler", "-s", type=str, default="gp",
                        choices=["random", "tpe", "gp", "all"],
                        help="Sampler to benchmark")

    parser.add_argument("--trials", "-t", type=int, default=ITERATIONS,
                        help="Number of trials/iterations")

    parser.add_argument("--jobs", "-j", type=int, default=THREADS,
                        help="Number of parallel jobs")

    parser.add_argument("--output", "-o", type=str, default=OUTPUT_DIR,
                        help="Output directory for results")

    args = parser.parse_args()

    # Print banner
    print("=" * 50)
    print("Optuna Benchmarking Tool")
    print(f"Started at: {time.strftime('%a %b %d %H:%M:%S %Y')}")
    print(f"Benchmark results will be saved to: {args.output}")

    # Generate synthetic data (matching HyperTune)
    print("\nGenerating synthetic dataset...")
    X_train, X_test, y_train, y_test = generate_synthetic_data(
        n_samples=1000,
        n_features=20,
        noise=0.1
    )
    print(f"Dataset generated: {X_train.shape[0]} training samples, "
          f"{X_test.shape[0]} test samples, {X_train.shape[1]} features.")

    # Define samplers to benchmark
    samplers = []
    if args.sampler == "all":
        samplers = ["random", "tpe"]
        if GP_AVAILABLE:
            samplers.append("gp")
    else:
        samplers = [args.sampler]

    # Run benchmarks
    for sampler_name in samplers:
        print(f"\n=== Benchmarking Optuna with {sampler_name} sampler ===")
        print(f"Iterations: {args.trials} | Threads: {args.jobs}")

        # Run benchmark
        result, study = run_benchmark(
            sampler_name,
            X_train, X_test, y_train, y_test,
            n_trials=args.trials,
            n_jobs=args.jobs
        )

        # Print summary
        print_benchmark_summary(result, sampler_name)

        # Save results
        csv_file, summary_file = save_benchmark_to_csv(
            f"{args.output}/{sampler_name}_benchmark.csv",
            result,
            sampler_name
        )

        print(f"Results saved to:")
        print(f" - {csv_file}")
        print(f" - {summary_file}")

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()

