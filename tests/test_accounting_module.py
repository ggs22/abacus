#!/usr/bin/env python3
"""
Comprehensive unit tests for the accounting module.

This module provides unit tests for:
- Account class
- AccountsList class  
- AccountFactory class
- Forecast strategies
- Utility functions

Tests are designed to be independent and use mocked data where appropriate
to avoid dependencies on actual account files.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

import pandas as pd
from omegaconf import DictConfig

# Import modules under test
from accounting import AccountFactory, PREDICTED_BALANCE
from accounting.Account import Account, AccountStats, print_codes_menu
from accounting.account_list import AccountsList, plot_forecast
from accounting.forecast_strategies import (
    ForecastStrategy, MonteCarloStrategy, ParallelMonteCarloStrategy,
    MeanTransactionsStrategy, PlannedTransactionsStrategy, NoTransactionsStrategy
)


class TestAccountClass(unittest.TestCase):
    """Test cases for the Account class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = DictConfig({
            'name': 'test_account',
            'account_dir': '/tmp/test_account',
            'status': 'OPEN',
            'columns': {
                'date': 'Date',
                'description': 'Description',
                'positive': ['Dépôts'],
                'negative': ['Retraits'],
                'balance': 'Solde'
            },
            'csv_format': {
                'separator': ',',
                'encoding': 'utf-8',
                'decimal': '.'
            }
        })
        
        # Sample transaction data
        self.sample_transactions = pd.DataFrame({
            'Date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
            'Description': ['Initial deposit', 'Withdrawal', 'Payment'],
            'Dépôts': [1000.0, 0.0, 0.0],
            'Retraits': [0.0, 50.0, 30.0],
            'Solde': [1000.0, 950.0, 920.0]
        })
        
    @patch('accounting.Account.Path')
    @patch('accounting.Account.pd.read_csv')
    def test_account_initialization(self, mock_read_csv, mock_path):
        """Test Account object initialization."""
        mock_read_csv.return_value = self.sample_transactions.copy()
        mock_path.return_value.exists.return_value = True
        
        account = Account(conf=self.test_config)
        
        self.assertEqual(account.name, 'test_account')
        self.assertEqual(account.status, 'OPEN')
        self.assertEqual(account.positive_names, ['Dépôts'])
        self.assertEqual(account.negative_names, ['Retraits'])
        self.assertEqual(account.balance_column_name, 'Solde')
        
    @patch('accounting.Account.Path')
    @patch('accounting.Account.pd.read_csv')
    def test_account_balance_calculation(self, mock_read_csv, mock_path):
        """Test balance calculation methods."""
        mock_read_csv.return_value = self.sample_transactions.copy()
        mock_path.return_value.exists.return_value = True
        
        account = Account(conf=self.test_config)
        account.transaction_data = self.sample_transactions.copy()
        account.transaction_data.set_index('Date', inplace=True)
        
        # Test current balance
        current_balance = account.get_balance()
        self.assertEqual(current_balance, 920.0)
        
        # Test balance at specific date
        balance_at_date = account.get_balance_at_date('2025-01-02')
        self.assertEqual(balance_at_date, 950.0)
        
    @patch('accounting.Account.Path')  
    @patch('accounting.Account.pd.read_csv')
    def test_account_transaction_filtering(self, mock_read_csv, mock_path):
        """Test transaction filtering and period stats."""
        mock_read_csv.return_value = self.sample_transactions.copy()
        mock_path.return_value.exists.return_value = True
        
        account = Account(conf=self.test_config)
        account.transaction_data = self.sample_transactions.copy()
        account.transaction_data.set_index('Date', inplace=True)
        
        # Add a 'code' column for testing
        account.transaction_data['code'] = ['deposit', 'withdrawal', 'payment']
        
        # Test period stats
        stats = account.period_stats(date='2025-01-03', last_n_days=2)
        self.assertIsInstance(stats, pd.DataFrame)
        
    def test_account_csv_column_validation(self):
        """Test CSV column name validation."""
        # Test with valid config
        account = Account(conf=self.test_config)
        self.assertTrue(hasattr(account, 'columns_names'))
        
    @patch('accounting.Account.Path')
    @patch('accounting.Account.json.load')
    def test_planned_transactions_loading(self, mock_json_load, mock_path):
        """Test planned transactions functionality."""
        mock_path.return_value.exists.return_value = True
        mock_json_load.return_value = [
            {
                'date': '2025-06-01',
                'description': 'Planned payment',
                'amount': -100.0,
                'code': 'planned'
            }
        ]
        
        account = Account(conf=self.test_config)
        planned = account.get_planned_transactions(
            start_date='2025-05-01', 
            predicted_days=60
        )
        
        if planned is not None:
            self.assertIsInstance(planned, pd.DataFrame)


class TestAccountsListClass(unittest.TestCase):
    """Test cases for the AccountsList class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock accounts
        self.mock_account1 = Mock(spec=Account)
        self.mock_account1.name = 'account1'
        self.mock_account1.balance_column_name = 'balance'
        self.mock_account1.color = 'blue'
        
        self.mock_account2 = Mock(spec=Account)
        self.mock_account2.name = 'account2'
        self.mock_account2.balance_column_name = 'balance'
        self.mock_account2.color = 'red'
        
        self.accounts_list = [self.mock_account1, self.mock_account2]
        
    def test_accounts_list_initialization(self):
        """Test AccountsList initialization."""
        acc_list = AccountsList(self.accounts_list)
        
        self.assertEqual(len(acc_list.accounts), 2)
        self.assertIn('account1', acc_list.accounts)
        self.assertIn('account2', acc_list.accounts)
        self.assertEqual(acc_list.accounts['account1'], self.mock_account1)
        
    def test_accounts_list_iteration(self):
        """Test AccountsList iteration capabilities."""
        acc_list = AccountsList(self.accounts_list)
        
        # Test if we can iterate over accounts
        account_names = []
        for account in acc_list.__acclist__:  # Use internal list for iteration
            account_names.append(account.name)
            
        self.assertEqual(set(account_names), {'account1', 'account2'})
        
    def test_accounts_list_indexing(self):
        """Test AccountsList indexing."""
        acc_list = AccountsList(self.accounts_list)
        
        # Test dictionary-style access
        self.assertEqual(acc_list['account1'], self.mock_account1)
        self.assertEqual(acc_list['account2'], self.mock_account2)


class TestAccountFactory(unittest.TestCase):
    """Test cases for the AccountFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.accounts_dir = Path(self.temp_dir)
        
    @patch('utils.path_utils.accounts_dir')
    def test_account_factory_initialization(self, mock_accounts_dir):
        """Test AccountFactory initialization."""
        mock_accounts_dir.return_value = self.accounts_dir
        
        factory = AccountFactory()
        self.assertIsInstance(factory, AccountFactory)
        
    @patch('utils.path_utils.accounts_dir')
    @patch('accounting.OmegaConf.load')
    @patch('accounting.Account')
    def test_load_accounts(self, mock_account_class, mock_omega_load, mock_accounts_dir):
        """Test loading accounts from directory."""
        mock_accounts_dir.return_value = self.accounts_dir
        
        # Create mock account directories
        test_account_dir = self.accounts_dir / 'test_account'
        test_account_dir.mkdir(parents=True)
        config_file = test_account_dir / 'config.yaml'
        config_file.touch()
        
        # Mock configuration loading
        mock_config = DictConfig({'name': 'test_account', 'status': 'OPEN'})
        mock_omega_load.return_value = mock_config
        
        # Mock Account creation
        mock_account = Mock(spec=Account)
        mock_account.name = 'test_account'
        mock_account_class.return_value = mock_account
        
        factory = AccountFactory()
        accounts_list = factory._load_accounts()
        
        self.assertIsInstance(accounts_list, AccountsList)


class TestForecastStrategies(unittest.TestCase):
    """Test cases for forecast strategy classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock account
        self.mock_account = Mock(spec=Account)
        self.mock_account.name = 'test_account'
        self.mock_account.status = 'OPEN'
        self.mock_account.columns_names = ['date', 'description', 'debit', 'credit', 'balance']
        self.mock_account.positive_names = ['credit']
        self.mock_account.negative_names = ['debit']
        self.mock_account.balance_column_name = 'balance'
        self.mock_account.most_recent_date = pd.Timestamp('2025-01-01')
        self.mock_account.serialized_self_path = '/tmp/test_account/account.pkl'
        
        # Mock account methods
        self.mock_account.get_planned_transactions.return_value = None
        self.mock_account.get_balance_at_date.return_value = 1000.0
        self.mock_account.period_stats.return_value = pd.DataFrame({
            'daily_prob': [0.1, 0.2],
            'mean': [-50.0, -30.0],
            'std': [10.0, 5.0],
            'daily_mean': [-5.0, -6.0]
        }, index=['grocery', 'transport'])
        
    def test_forecast_strategy_base_class(self):
        """Test ForecastStrategy base class functionality."""
        strategy = ForecastStrategy()
        
        # Test path generation
        path = strategy.get_serialized_prediction_path(self.mock_account, '2025-01-01')
        self.assertIsInstance(path, Path)
        self.assertTrue(path.name.endswith('_prediction.pkl'))
        
    @patch('accounting.forecast_strategies.Path')
    def test_mean_transactions_strategy(self, mock_path):
        """Test MeanTransactionsStrategy."""
        mock_path.return_value.exists.return_value = False
        
        strategy = MeanTransactionsStrategy()
        
        # Mock the prediction wrapper to avoid file I/O
        with patch.object(strategy, '_prediction_wraper') as mock_wrapper:
            mock_result = pd.DataFrame({
                'date': pd.to_datetime(['2025-01-02', '2025-01-03']),
                'description': ['pred1', 'pred2'],
                'debit': [0, 30],
                'credit': [0, 0],
                'balance': [1000, 970]
            })
            mock_wrapper.return_value = mock_result
            
            result = strategy.predict(
                account=self.mock_account,
                predicted_days=2
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            mock_wrapper.assert_called_once()
            
    @patch('accounting.forecast_strategies.Path')
    def test_no_transactions_strategy(self, mock_path):
        """Test NoTransactionsStrategy."""
        mock_path.return_value.exists.return_value = False
        
        strategy = NoTransactionsStrategy()
        
        with patch.object(strategy, '_prediction_wraper') as mock_wrapper:
            mock_result = pd.DataFrame({
                'date': pd.to_datetime(['2025-01-02', '2025-01-03']),
                'description': ['pred1', 'pred2'],
                'debit': [0, 0],
                'credit': [0, 0],
                'balance': [1000, 1000]
            })
            mock_wrapper.return_value = mock_result
            
            result = strategy.predict(
                account=self.mock_account,
                predicted_days=2,
                balance_offset=0
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            
    @patch('accounting.forecast_strategies.Path')
    def test_monte_carlo_strategy(self, mock_path):
        """Test MonteCarloStrategy."""
        mock_path.return_value.exists.return_value = False
        
        strategy = MonteCarloStrategy()
        
        with patch.object(strategy, '_prediction_wraper') as mock_wrapper:
            mock_result = pd.DataFrame({
                'date': pd.to_datetime(['2025-01-02'] * 4),
                'description': ['pred1'] * 4,
                'debit': [0, 30, 0, 25],
                'credit': [0, 0, 0, 0],
                'balance': [1000, 970, 1000, 975],
                'iteration': [0, 0, 1, 1]
            })
            mock_wrapper.return_value = mock_result
            
            result = strategy.predict(
                account=self.mock_account,
                predicted_days=1,
                mc_iterations=2
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            
    @patch('accounting.forecast_strategies.Path')
    def test_parallel_monte_carlo_strategy(self, mock_path):
        """Test ParallelMonteCarloStrategy."""
        mock_path.return_value.exists.return_value = False
        
        strategy = ParallelMonteCarloStrategy(max_workers=2)
        
        with patch.object(strategy, '_prediction_wraper') as mock_wrapper:
            mock_result = pd.DataFrame({
                'date': pd.to_datetime(['2025-01-02'] * 4),
                'description': ['pred1'] * 4,
                'debit': [0, 30, 0, 25],
                'credit': [0, 0, 0, 0],
                'balance': [1000, 970, 1000, 975],
                'iteration': [0, 0, 1, 1]
            })
            mock_wrapper.return_value = mock_result
            
            result = strategy.predict(
                account=self.mock_account,
                predicted_days=1,
                mc_iterations=2
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(strategy.max_workers, 2)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions in the accounting module."""
    
    def test_print_codes_menu(self):
        """Test print_codes_menu function."""
        codes = ['grocery', 'transport', 'entertainment']
        transaction = pd.Series({
            'date': '2025-01-01',
            'description': 'Test transaction',
            'amount': -50.0
        })
        
        # Test that function runs without error
        # We can't easily test the output without mocking print
        with patch('builtins.print') as mock_print:
            print_codes_menu(codes, transaction)
            # Verify that print was called (menu was displayed)
            self.assertTrue(mock_print.called)
            
    def test_predicted_balance_constant(self):
        """Test that PREDICTED_BALANCE constant is properly defined."""
        from accounting.Account import PREDICTED_BALANCE
        self.assertIsInstance(PREDICTED_BALANCE, str)
        self.assertEqual(PREDICTED_BALANCE, 'predicted_balance')


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios combining multiple components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_account_with_forecast_integration(self):
        """Test Account class integration with forecast strategies."""
        # Create a mock account with realistic data
        config = DictConfig({
            'name': 'integration_test',
            'account_dir': self.temp_dir,
            'status': 'OPEN',
            'columns': {
                'date': 'Date',
                'description': 'Description', 
                'positive': ['Credit'],
                'negative': ['Debit'],
                'balance': 'Balance'
            }
        })
        
        with patch('accounting.Account.Path'), \
             patch('accounting.Account.pd.read_csv') as mock_read_csv:
            
            # Mock transaction data
            transaction_data = pd.DataFrame({
                'Date': pd.to_datetime(['2025-01-01', '2025-01-02']),
                'Description': ['Deposit', 'Withdrawal'],
                'Credit': [1000.0, 0.0],
                'Debit': [0.0, 100.0],
                'Balance': [1000.0, 900.0]
            })
            mock_read_csv.return_value = transaction_data
            
            account = Account(conf=config)
            
            # Test that account can be used with forecast strategy
            strategy = NoTransactionsStrategy()
            
            with patch.object(strategy, '_prediction_wraper') as mock_wrapper:
                mock_wrapper.return_value = pd.DataFrame({
                    'Date': pd.to_datetime(['2025-01-03']),
                    'Description': ['Predicted'],
                    'Credit': [0.0],
                    'Debit': [0.0],
                    'Balance': [900.0]
                })
                
                result = strategy.predict(
                    account=account,
                    predicted_days=1
                )
                
                self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)