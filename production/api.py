"""
Production FastAPI Service for RL Portfolio Optimization.

Provides RESTful API for portfolio recommendations, risk monitoring,
and client reporting.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RL Portfolio Optimization API",
    description="Production API for AI-powered portfolio management",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# ============================================================================
# Pydantic Models
# ============================================================================


class PortfolioRequest(BaseModel):
    """Request model for portfolio recommendations."""

    client_id: str = Field(..., description="Unique client identifier")
    risk_tolerance: str = Field(..., description="Risk tolerance: low, medium, high")
    investment_amount: float = Field(..., gt=0, description="Investment amount in USD")
    constraints: Optional[Dict] = Field(None, description="Custom constraints")


class PortfolioRecommendation(BaseModel):
    """Response model for portfolio recommendations."""

    client_id: str
    timestamp: datetime
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    confidence_score: float


class RiskAlert(BaseModel):
    """Model for risk monitoring alerts."""

    alert_id: str
    client_id: str
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    current_value: float
    threshold: float


class PerformanceMetrics(BaseModel):
    """Model for portfolio performance metrics."""

    client_id: str
    period: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


# ============================================================================
# Utility Functions
# ============================================================================


def load_model(model_name: str):
    """Load trained RL model."""
    model_path = Path(config["output"]["models_dir"]) / f"{model_name}.zip"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    try:
        # Load model based on algorithm
        if model_name == "ppo":
            from stable_baselines3 import PPO

            model = PPO.load(str(model_path))
        elif model_name == "sac":
            from stable_baselines3 import SAC

            model = SAC.load(str(model_path))
        # Add other models as needed

        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Error loading model")


def get_market_data(assets: List[str], days: int = 60) -> pd.DataFrame:
    """Fetch recent market data."""
    # In production, this would fetch from a database or API
    # For now, return placeholder
    logger.info(f"Fetching market data for {len(assets)} assets")
    return pd.DataFrame()


def calculate_risk_metrics(weights: np.ndarray, returns_data: pd.DataFrame) -> Dict:
    """Calculate portfolio risk metrics."""
    mean_returns = returns_data.mean()
    cov_matrix = returns_data.cov()

    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(
        252
    )
    sharpe = (portfolio_return - 0.045) / portfolio_vol if portfolio_vol > 0 else 0

    return {
        "expected_return": float(portfolio_return * 100),
        "expected_volatility": float(portfolio_vol * 100),
        "sharpe_ratio": float(sharpe),
    }


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RL Portfolio Optimization API",
        "version": "1.0.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/portfolio/recommend", response_model=PortfolioRecommendation)
async def get_portfolio_recommendation(request: PortfolioRequest):
    """
    Get portfolio weight recommendations using trained RL agent.

    Args:
        request: Portfolio recommendation request

    Returns:
        Portfolio weights and expected metrics
    """
    try:
        logger.info(f"Processing recommendation for client {request.client_id}")

        # Select model based on risk tolerance
        model_map = {
            "low": "qr_ddpg",  # Best for tail-risk management
            "medium": "ppo",  # Best overall Sharpe
            "high": "sac",  # More exploratory
        }

        model_name = model_map.get(request.risk_tolerance, "ppo")
        model = load_model(model_name)

        # Get current market state
        assets = config["data"]["assets"]
        all_tickers = sum(assets.values(), [])
        market_data = get_market_data(all_tickers)

        # Get model prediction
        # In production, construct proper state from market_data
        state = np.zeros(100)  # Placeholder

        action, _ = model.predict(state, deterministic=True)
        weights = np.clip(action, 0, 1)
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        # Calculate risk metrics
        returns_data = market_data  # Placeholder
        risk_metrics = (
            calculate_risk_metrics(weights, returns_data)
            if len(returns_data) > 0
            else {}
        )

        # Create weight dictionary
        weight_dict = {ticker: float(w) for ticker, w in zip(all_tickers, weights)}

        return PortfolioRecommendation(
            client_id=request.client_id,
            timestamp=datetime.now(),
            weights=weight_dict,
            expected_return=risk_metrics.get("expected_return", 0),
            expected_volatility=risk_metrics.get("expected_volatility", 0),
            sharpe_ratio=risk_metrics.get("sharpe_ratio", 0),
            max_drawdown_estimate=-15.0,  # Placeholder
            confidence_score=0.85,
        )

    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/performance/{client_id}", response_model=PerformanceMetrics)
async def get_portfolio_performance(client_id: str, period: str = "1M"):
    """
    Get portfolio performance metrics for a client.

    Args:
        client_id: Client identifier
        period: Time period (1D, 1W, 1M, 3M, 6M, 1Y)

    Returns:
        Performance metrics
    """
    try:
        # In production, fetch from database
        logger.info(f"Fetching performance for client {client_id}, period {period}")

        return PerformanceMetrics(
            client_id=client_id,
            period=period,
            total_return=12.5,
            annualized_return=25.3,
            volatility=15.2,
            sharpe_ratio=1.85,
            max_drawdown=-8.5,
            win_rate=0.65,
        )

    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/risk/monitor/{client_id}", response_model=List[RiskAlert])
async def monitor_portfolio_risk(client_id: str):
    """
    Monitor portfolio for risk violations.

    Args:
        client_id: Client identifier

    Returns:
        List of active risk alerts
    """
    try:
        logger.info(f"Monitoring risk for client {client_id}")

        alerts = []

        # Check drawdown
        current_dd = -12.5
        max_dd_threshold = (
            config["production"]["risk_monitoring"]["max_drawdown_alert"] * 100
        )

        if abs(current_dd) > max_dd_threshold:
            alerts.append(
                RiskAlert(
                    alert_id=f"DD_{client_id}_{datetime.now().timestamp()}",
                    client_id=client_id,
                    timestamp=datetime.now(),
                    alert_type="MAX_DRAWDOWN",
                    severity="WARNING",
                    message=f"Drawdown exceeded threshold: {current_dd:.1f}% > {max_dd_threshold:.1f}%",
                    current_value=current_dd,
                    threshold=-max_dd_threshold,
                )
            )

        return alerts

    except Exception as e:
        logger.error(f"Error in risk monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rebalance/{client_id}")
async def trigger_rebalance(client_id: str, background_tasks: BackgroundTasks):
    """
    Trigger portfolio rebalancing for a client.

    Args:
        client_id: Client identifier
        background_tasks: FastAPI background tasks

    Returns:
        Rebalancing status
    """
    try:
        logger.info(f"Triggering rebalance for client {client_id}")

        # Add rebalancing task to background
        background_tasks.add_task(execute_rebalance, client_id)

        return {
            "status": "pending",
            "client_id": client_id,
            "message": "Rebalancing scheduled",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error triggering rebalance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/list")
async def list_available_models():
    """List all available trained models."""
    try:
        models_dir = Path(config["output"]["models_dir"])
        models = [f.stem for f in models_dir.glob("*.zip")]

        return {"models": models, "count": len(models)}

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Background Tasks
# ============================================================================


async def execute_rebalance(client_id: str):
    """Execute portfolio rebalancing in background."""
    logger.info(f"Executing rebalance for {client_id}")
    # Implementation here


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting RL Portfolio Optimization API")
    logger.info(f"Configuration loaded: {len(config)} sections")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=config["production"]["api"]["host"],
        port=config["production"]["api"]["port"],
        reload=True,
    )
