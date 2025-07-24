#!/usr/bin/env python3
"""
Performance benchmark comparing original MonteCarloStrategy with ParallelMonteCarloStrategy.

This script demonstrates the performance improvements achieved through parallelization
of Monte Carlo simulations for financial forecasting.
"""

import time
from pathlib import Path
import sys

# Add the accounting module to path
sys.path.append(str(Path(__file__).parent))

from accounting.forecast_strategies import MonteCarloStrategy, ParallelMonteCarloStrategy
from accounting import AccountFactory


def benchmark_monte_carlo_strategies():
    """
    Benchmark both Monte Carlo strategies and compare performance.
    """
    print("Monte Carlo Strategy Performance Benchmark")
    print("=" * 50)
    
    # Load a sample account for testing
    try:
        account_factory = AccountFactory()
        # Try to load any available account
        available_accounts = list(Path("accounting/accounts").iterdir())
        if not available_accounts:
            print("No accounts found for benchmarking.")
            return
            
        account_name = available_accounts[0].name
        account = account_factory.load_account(account_name)
        
        if account is None:
            print(f"Could not load account: {account_name}")
            return
            
    except Exception as e:
        print(f"Error loading account: {e}")
        return
    
    # Benchmark parameters
    test_configs = [
        {"mc_iterations": 50, "predicted_days": 30},
        {"mc_iterations": 100, "predicted_days": 60},
        {"mc_iterations": 200, "predicted_days": 90},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting with {config['mc_iterations']} iterations, {config['predicted_days']} days")
        print("-" * 30)
        
        # Test original MonteCarloStrategy
        original_strategy = MonteCarloStrategy()
        start_time = time.time()
        
        try:
            original_result = original_strategy.predict(
                account=account,
                predicted_days=config['predicted_days'],
                force_new=True,
                **config
            )
            original_time = time.time() - start_time
            original_rows = len(original_result) if original_result is not None else 0
            print(f"Original Strategy: {original_time:.2f}s ({original_rows} rows)")
            
        except Exception as e:
            print(f"Original Strategy failed: {e}")
            original_time = float('inf')
            original_rows = 0
        
        # Test ParallelMonteCarloStrategy
        parallel_strategy = ParallelMonteCarloStrategy()
        start_time = time.time()
        
        try:
            parallel_result = parallel_strategy.predict(
                account=account,
                predicted_days=config['predicted_days'],
                force_new=True,
                **config
            )
            parallel_time = time.time() - start_time
            parallel_rows = len(parallel_result) if parallel_result is not None else 0
            print(f"Parallel Strategy: {parallel_time:.2f}s ({parallel_rows} rows)")
            
            # Calculate speedup
            if original_time > 0 and original_time != float('inf'):
                speedup = original_time / parallel_time
                print(f"Speedup: {speedup:.2f}x")
            else:
                speedup = 0
                
        except Exception as e:
            print(f"Parallel Strategy failed: {e}")
            parallel_time = float('inf')
            parallel_rows = 0
            speedup = 0
        
        results.append({
            'config': config,
            'original_time': original_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'original_rows': original_rows,
            'parallel_rows': parallel_rows
        })
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    for result in results:
        config = result['config']
        print(f"\nConfig: {config['mc_iterations']} iterations, {config['predicted_days']} days")
        print(f"  Original: {result['original_time']:.2f}s")
        print(f"  Parallel: {result['parallel_time']:.2f}s")
        if result['speedup'] > 0:
            print(f"  Speedup: {result['speedup']:.2f}x")
        else:
            print("  Speedup: N/A (benchmark failed)")
    
    # Calculate average speedup
    valid_speedups = [r['speedup'] for r in results if r['speedup'] > 0]
    if valid_speedups:
        avg_speedup = sum(valid_speedups) / len(valid_speedups)
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    print("\nNotes:")
    print("- Parallel strategy uses multiprocessing to distribute Monte Carlo iterations")
    print("- Performance gains depend on CPU core count and workload size")
    print("- Results include both computation and data serialization time")


if __name__ == "__main__":
    benchmark_monte_carlo_strategies()