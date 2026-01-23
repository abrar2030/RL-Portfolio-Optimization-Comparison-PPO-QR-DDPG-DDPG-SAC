## Deep Reinforcement Learning for Portfolio Management

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)](production/api.py)

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Docker Deployment](#-docker-deployment)
- [New Features Guide](#-new-features-guide)
- [API Documentation](#-api-documentation)
- [Configuration](#%EF%B8%8F-configuration)
- [Project Structure](#-project-structure)
- [Results & Analysis](#-results--analysis)

---

### ✨ Key Features

1. **🐳 Containerization & Deployment**
   - Multi-stage Dockerfile with GPU support
   - Docker Compose orchestration (API, training, monitoring, database)
   - Pre-configured for production deployment

2. **💰 Transaction Cost Analysis**
   - Multiple cost structures (retail 0.5%, institutional 0.05%, zero-cost baseline)
   - Optimal rebalancing frequency analysis
   - Detailed cost impact visualization and reporting

3. **🧪 Reward Function Ablation Study**
   - Systematic testing of λ (drawdown penalty) from 0.0 to 1.0
   - Performance surface visualization
   - Statistical analysis of optimal parameter ranges

4. **⚖️ Market Constraints**
   - Short-selling controls
   - Leverage limits (configurable max leverage)
   - Position size limits (per-asset caps)
   - Sector exposure limits

5. **📊 Market Regime Analysis**
   - Bull/Bear/Sideways regime classification
   - Algorithm performance breakdown by regime
   - Best-strategy recommendations per market condition

6. **🏭 Production Deployment Architecture**
   - FastAPI REST API for predictions
   - Scheduled rebalancing with Celery
   - Real-time risk monitoring
   - PostgreSQL for data persistence
   - Redis for caching
   - Grafana dashboards

7. **📈 Extended Benchmarks**
   - 60/40 Portfolio (classic balanced)
   - All-Weather Portfolio (Ray Dalio)
   - Minimum Correlation Portfolio
   - Original 5 benchmarks (MVO, Risk-Parity, etc.)

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/quantsingularity/RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC
cd RL-Portfolio-Optimization-Comparison-PPO-QR-DDPG-DDPG-SAC

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# Access services
# - API: http://localhost:8000
# - Jupyter: http://localhost:8888
# - Grafana: http://localhost:3000
```

### Option 2: Local Installation

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt

# Run data preparation
python code/data_processor.py

# Train agents
python code/train.py

# Run evaluation
python code/evaluate.py

# Generate figures
python code/figure_generation.py
```

---

## 🐳 Docker Deployment

### Services Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│   PostgreSQL    │     │   Redis Cache   │
│   (Port 8000)   │     │   (Port 5432)   │     │   (Port 6379)   │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         │              ┌─────────────────┐
         └─────────────▶│  Celery Worker  │
                        │  (Background)   │
                        └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │  Training GPU   │
                        │  (CUDA Support) │
                        └─────────────────┘
```

### GPU Support

```bash
# Enable GPU training
docker-compose up training

# Check GPU availability
docker exec -it rl-portfolio-training nvidia-smi
```

### Environment Variables

Create `.env` file:

```bash
POSTGRES_USER=portfolio_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=portfolio_db
GRAFANA_PASSWORD=admin
CUDA_VISIBLE_DEVICES=0
```

---

## 🎯 New Features Guide

### 1. Transaction Cost Analysis

Analyze portfolio performance under different transaction cost structures:

```python
from code.transaction_cost_analysis import TransactionCostAnalyzer

analyzer = TransactionCostAnalyzer()

# Analyze with different cost structures
results = analyzer.analyze_rebalancing_frequency(
    strategy_name='ppo',
    portfolio_values_base=portfolio_values,
    portfolio_weights_history=weights_history,
    dates=dates
)

# Generate visualization
analyzer.plot_cost_impact(results, save_path='results/cost_analysis.png')

# Generate report
report = analyzer.generate_cost_report(results, 'results/cost_report.txt')
```

**Key Insights:**

- Retail costs (0.5%) can reduce returns by 15-20%
- Optimal rebalancing: **Weekly to Biweekly** for most strategies
- Institutional costs (0.05%) have minimal impact (<2%)

### 2. Reward Ablation Study

Systematically test λ parameter impact:

```python
from code.reward_ablation import RewardAblationStudy

study = RewardAblationStudy()

# Run ablation across λ ∈ [0, 1]
results = study.run_ablation_study(
    agent_class=PPO,
    env_factory=create_env_with_lambda,
    training_steps=100000
)

# Analyze results
analysis = study.analyze_results(results)
print(f"Optimal λ for Sharpe: {analysis['optimal_lambda_sharpe']}")

# Visualize
study.plot_performance_surface(results, 'results/ablation_surface.png')
```

**Finding:** Optimal λ = 0.5 balances returns and risk control.

### 3. Market Regime Analysis

Break down performance by market conditions:

```python
from code.regime_analysis import MarketRegimeAnalyzer

analyzer = MarketRegimeAnalyzer()

# Identify regimes using VIX
regime_df = analyzer.identify_regimes_vix(market_data)

# Analyze performance
performance = analyzer.analyze_performance_by_regime(
    strategy_results={'ppo': ppo_results, 'qr_ddpg': qr_results},
    regime_labels=regime_df
)

# Find best strategies per regime
best = analyzer.compare_algorithms_by_regime(performance)
print(f"Bull market: {best['bull']['strategy']}")
print(f"Bear market: {best['bear']['strategy']}")

# Visualize
analyzer.plot_regime_performance(performance, 'results/regime_analysis.png')
```

**Results:**

- **Bull Markets**: SAC performs best (Sharpe: 2.5+)
- **Bear Markets**: QR-DDPG excels (lowest CVaR)
- **Sideways**: PPO provides consistent performance

### 4. Extended Benchmarks

Compare against additional traditional strategies:

```python
from code.enhanced_benchmarks import EnhancedBenchmarkStrategies

strategies = EnhancedBenchmarkStrategies(returns_data, tickers, asset_classes)

# 60/40 Portfolio
weights_60_40 = strategies.sixty_forty()

# All-Weather Portfolio (Ray Dalio)
weights_all_weather = strategies.all_weather()

# Minimum Correlation
weights_min_corr = strategies.minimum_correlation()
```

---

## 📡 API Documentation

### Start API Server

```bash
# Using Docker
docker-compose up api

# Using Python
python production/api.py
```

API available at: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Example API Calls

#### Get Portfolio Recommendation

```bash
curl -X POST "http://localhost:8000/api/v1/portfolio/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "client_001",
    "risk_tolerance": "medium",
    "investment_amount": 1000000
  }'
```

Response:

```json
{
  "client_id": "client_001",
  "timestamp": "2024-01-15T10:30:00",
  "weights": {
    "AAPL": 0.12,
    "MSFT": 0.10,
    "BTC-USD": 0.08,
    ...
  },
  "expected_return": 25.3,
  "expected_volatility": 18.5,
  "sharpe_ratio": 1.85
}
```

#### Monitor Portfolio Risk

```bash
curl "http://localhost:8000/api/v1/risk/monitor/client_001"
```

#### Get Performance Metrics

```bash
curl "http://localhost:8000/api/v1/portfolio/performance/client_001?period=1M"
```

---

## ⚙️ Configuration

All settings in `config/config.yaml`:

```yaml
# Transaction Costs
transaction_costs:
  cost_structures:
    retail: 0.005
    standard: 0.001
    institutional: 0.0005
  rebalance_frequencies:
    daily: 1
    weekly: 5
    monthly: 20

# Market Constraints
constraints:
  short_selling: false
  max_leverage: 1.0
  max_position_size: 0.3
  sector_exposure_limits:
    equities: 0.6
    cryptocurrencies: 0.3

# Production Deployment
production:
  api:
    host: "0.0.0.0"
    port: 8000
  rebalancing:
    frequency: "weekly"
    time: "09:30"
  risk_monitoring:
    max_drawdown_alert: 0.15
```

---

## 📂 Project Structure

```
rl-portfolio/
├── Dockerfile                          # Multi-stage container
├── docker-compose.yml                  # Service orchestration
├── requirements.txt                    # Core dependencies
├── requirements-prod.txt               # Production dependencies
│
├── code/                              # Core implementation
│   ├── data_processor.py              # Data pipeline
│   ├── environment.py                 # RL environment
│   ├── agents.py                      # DRL agents
│   ├── benchmarks.py                  # Original benchmarks
│   ├── enhanced_benchmarks.py         # 🆕 Extended benchmarks
│   ├── transaction_cost_analysis.py   # 🆕 Cost analysis
│   ├── reward_ablation.py             # 🆕 Ablation study
│   ├── regime_analysis.py             # 🆕 Regime analysis
│   ├── train.py                       # Training pipeline
│   └── evaluate.py                    # Evaluation pipeline
│
├── production/                        # 🆕 Production services
│   ├── api.py                         # FastAPI service
│   ├── tasks.py                       # Celery tasks
│   ├── models.py                      # Database models
│   └── monitoring.py                  # Risk monitoring
│
├── config/
│   └── config.yaml                    # Enhanced configuration
│
├── notebooks/                         # 🆕 Jupyter notebooks
│   ├── transaction_cost_analysis.ipynb
│   ├── reward_ablation.ipynb
│   └── regime_analysis.ipynb
│
├── tests/                             # Unit tests
│   └── test_all.py
│
└── results/                           # Output directory
    ├── figures/                       # Visualizations
    ├── reports/                       # PDF/HTML reports
    └── logs/                          # Training logs
```

---

## 📊 Results & Analysis

### Performance Comparison (Test Period: 2023-2024)

| Strategy        | Sharpe | Annual Return | Max DD | CVaR 5% | Cost Impact |
| --------------- | ------ | ------------- | ------ | ------- | ----------- |
| **PPO**         | 2.15   | 38.2%         | -7.2%  | -1.8%   | -2.1%       |
| **QR-DDPG**     | 2.08   | 36.5%         | -6.5%  | -1.5%   | -1.8%       |
| **SAC**         | 1.98   | 35.1%         | -8.8%  | -2.1%   | -2.3%       |
| 60/40 Portfolio | 1.52   | 22.5%         | -11.2% | -3.2%   | -0.8%       |
| All-Weather     | 1.65   | 24.1%         | -9.5%  | -2.8%   | -0.6%       |
| Risk-Parity     | 1.45   | 25.8%         | -12.1% | -3.1%   | -1.2%       |

### Transaction Cost Impact

- **Daily Rebalancing**: -5.2% annual return impact
- **Weekly Rebalancing**: -1.8% annual return impact ✅ Optimal
- **Monthly Rebalancing**: -0.9% impact but higher variance

### Regime Performance

| Algorithm | Bull Sharpe | Bear Sharpe | Sideways Sharpe |
| --------- | ----------- | ----------- | --------------- |
| PPO       | 2.3         | 1.8         | 2.0             |
| QR-DDPG   | 2.1         | **2.2**     | 1.9             |
| SAC       | **2.5**     | 1.5         | 1.8             |

---

## 🔧 Production Deployment

### Scheduled Rebalancing

```python
# Celery beat schedule (production/tasks.py)
@celery_app.task
def scheduled_rebalance():
    """Execute daily rebalancing at market open."""
    clients = get_active_clients()
    for client in clients:
        generate_and_execute_rebalance(client)
```

### Risk Monitoring

```python
@celery_app.task
def monitor_portfolio_risks():
    """Monitor all portfolios for risk violations."""
    for portfolio in active_portfolios():
        if portfolio.drawdown > MAX_DRAWDOWN_THRESHOLD:
            send_risk_alert(portfolio)
            reduce_position_sizes(portfolio)
```

### Client Reporting

```bash
# Generate client reports
python production/generate_reports.py --period weekly --format pdf
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.
