#!/usr/bin/env python3
"""
Example usage of the new ParallelMonteCarloStrategy.

This script demonstrates how to use the parallelized Monte Carlo forecasting
strategy as a drop-in replacement for the original implementation.
"""

import sys
from pathlib import Path

# Add the accounting module to path
sys.path.append(str(Path(__file__).parent))

from accounting.forecast_strategies import ParallelMonteCarloStrategy
from accounting import AccountFactory


def main():
    """
    Demonstrate usage of ParallelMonteCarloStrategy.
    """
    print("Parallel Monte Carlo Strategy Example")
    print("=" * 40)
    
    # Initialize the strategy with custom worker count (optional)
    # If max_workers is not specified, it will use all available CPU cores
    parallel_strategy = ParallelMonteCarloStrategy(max_workers=4)
    
    try:
        # Load an account
        account_factory = AccountFactory()
        available_accounts = list(Path("accounting/accounts").iterdir())
        
        if not available_accounts:
            print("No accounts found. Please ensure account data is available.")
            return
            
        account_name = available_accounts[0].name
        account = account_factory.load_account(account_name)
        
        if account is None:
            print(f"Could not load account: {account_name}")
            return
            
        print(f"Loaded account: {account.name}")
        
        # Generate forecast using parallel Monte Carlo
        print("\\nGenerating parallel Monte Carlo forecast...")
        
        forecast_result = parallel_strategy.predict(
            account=account,
            predicted_days=60,          # Forecast 60 days into the future
            mc_iterations=500,          # Run 500 Monte Carlo iterations
            force_new=True,             # Force new calculation (ignore cache)
            balance_offset=0            # No initial balance offset
        )
        
        if forecast_result is not None:
            print(f"Forecast generated successfully!")
            print(f"Result shape: {forecast_result.shape}")
            print(f"Columns: {list(forecast_result.columns)}")
            print(f"Date range: {forecast_result['date'].min()} to {forecast_result['date'].max()}")
            print(f"Iteration range: {forecast_result['iteration'].min()} to {forecast_result['iteration'].max()}")
            
            # Show sample statistics
            if account.balance_column_name in forecast_result.columns:
                balance_stats = forecast_result[account.balance_column_name].describe()
                print(f"\\nBalance forecast statistics:")
                print(f"  Mean: ${balance_stats['mean']:.2f}")
                print(f"  Std Dev: ${balance_stats['std']:.2f}")
                print(f"  Min: ${balance_stats['min']:.2f}")
                print(f"  Max: ${balance_stats['max']:.2f}")
        else:
            print("Forecast generation failed.")
            
    except Exception as e:
        print(f"Error during forecast generation: {e}")
        import traceback
        traceback.print_exc()


def compare_strategies_example():
    """
    Example showing how to compare original and parallel strategies.
    """
    print("\\n" + "=" * 40)
    print("Strategy Comparison Example")
    print("=" * 40)
    
    from accounting.forecast_strategies import MonteCarloStrategy
    
    try:
        # Load account
        account_factory = AccountFactory()
        available_accounts = list(Path("accounting/accounts").iterdir())
        
        if not available_accounts:
            return
            
        account = account_factory.load_account(available_accounts[0].name)
        if account is None:
            return
        
        # Test parameters
        test_params = {
            'predicted_days': 30,
            'mc_iterations': 100,
            'force_new': True
        }
        
        print(f"Comparing strategies with {test_params['mc_iterations']} iterations...")
        
        # Original strategy
        original_strategy = MonteCarloStrategy()
        original_result = original_strategy.predict(account=account, **test_params)
        
        # Parallel strategy  
        parallel_strategy = ParallelMonteCarloStrategy()
        parallel_result = parallel_strategy.predict(account=account, **test_params)
        
        # Compare results
        if original_result is not None and parallel_result is not None:
            print(f"Original result shape: {original_result.shape}")
            print(f"Parallel result shape: {parallel_result.shape}")
            
            # Check if results are consistent (they should be similar statistically)
            orig_balance_mean = original_result[account.balance_column_name].mean()
            parallel_balance_mean = parallel_result[account.balance_column_name].mean()
            
            print(f"Original mean balance: ${orig_balance_mean:.2f}")
            print(f"Parallel mean balance: ${parallel_balance_mean:.2f}")
            print(f"Difference: ${abs(orig_balance_mean - parallel_balance_mean):.2f}")
            
    except Exception as e:
        print(f"Comparison failed: {e}")


if __name__ == "__main__":
    main()
    compare_strategies_example()