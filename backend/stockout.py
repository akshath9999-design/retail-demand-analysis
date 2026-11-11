"""
backend/stockout.py
Stockout Risk Analysis and Inventory Optimization Engine.
Clearly separates ML demand forecasting from rule-based risk classification
and replenishment quantity computation.
"""

from typing import Dict, Any, Tuple
from backend.prediction import forecaster


def compute_safety_stock_buffer(
    lead_time_demand: float,
    safety_stock_threshold_pct: float
) -> float:
    """Computes safety stock buffer = lead-time demand * configured safety buffer percentage."""
    if safety_stock_threshold_pct < 0.0 or safety_stock_threshold_pct > 1.0:
        raise ValueError("safety_stock_threshold_pct must be between 0.0 and 1.0.")
    if lead_time_demand <= 0.0:
        return 0.0
    return round(float(lead_time_demand) * float(safety_stock_threshold_pct), 1)


def classify_stockout_risk(
    current_inventory: int,
    lead_time_demand: float,
    safety_stock_buffer: float
) -> Tuple[str, float, str]:
    """
    Rule-based stockout risk classifier.

    Rule:
      Remaining Inventory = current_inventory - lead_time_demand
      Risk = HIGH if remaining_inventory < safety_stock_buffer (strictly below threshold),
             otherwise LOW (remaining_inventory >= safety_stock_buffer).

    Returns:
      (stockout_status, remaining_inventory, risk_color)
    """
    remaining_inventory = round(float(current_inventory) - float(lead_time_demand), 1)
    if remaining_inventory < round(float(safety_stock_buffer), 1):
        return "HIGH", remaining_inventory, "#dc3545"
    return "LOW", remaining_inventory, "#28a745"


def compute_replenishment_quantity(
    stockout_status: str,
    current_inventory: int,
    lead_time_demand: float,
    safety_stock_buffer: float
) -> int:
    """
    Calculates suggested replenishment order quantity.
      - If HIGH risk: max(0, round(lead_time_demand + safety_stock_buffer - current_inventory))
      - If LOW risk:  0
    """
    if stockout_status != "HIGH":
        return 0
    return max(0, int(round((float(lead_time_demand) + float(safety_stock_buffer)) - float(current_inventory))))


def analyze_stockout_risk(
    stock_code: str,
    current_inventory: int,
    supplier_lead_time: int,
    safety_stock_threshold_pct: float = 0.20
) -> Dict[str, Any]:
    """
    Evaluates stockout risk based on ML-forecasted demand over the supplier lead time.

    Explicit Rule & Assumptions:
      - Lead-Time Demand (D_lead) = Sum of ML-predicted daily demand for `supplier_lead_time` days.
      - Safety Stock Buffer (SS) = D_lead * safety_stock_threshold_pct.
      - Remaining Inventory = current_inventory - D_lead.
      - Risk = HIGH if Remaining Inventory < SS, else LOW.
      - Replenishment = max(0, round(D_lead + SS - current_inventory)) if HIGH, else 0.
      - Assumes user-supplied `current_inventory` since UCI Online Retail lacks warehouse stock records.
    """
    stock_code = stock_code.strip().upper()
    if not isinstance(supplier_lead_time, int) or isinstance(supplier_lead_time, bool) or supplier_lead_time < 1:
        raise ValueError("Supplier lead time must be an integer >= 1 day.")
    if current_inventory < 0:
        raise ValueError("Available inventory cannot be negative.")
    if safety_stock_threshold_pct < 0.0 or safety_stock_threshold_pct > 1.0:
        raise ValueError("Safety stock threshold percentage must be between 0.0 and 1.0.")

    # 1. ML Demand Forecasting Step
    forecast_result = forecaster.forecast(stock_code, forecast_days=supplier_lead_time)
    lead_time_demand = float(forecast_result['total_forecast_demand'])
    avg_daily_demand = float(forecast_result['avg_daily_demand'])
    desc = forecast_result['description']

    # 2. Rule-Based Safety Stock & Risk Classification Step
    safety_stock_buffer = compute_safety_stock_buffer(lead_time_demand, safety_stock_threshold_pct)
    stockout_status, remaining_inventory, risk_color = classify_stockout_risk(
        current_inventory=current_inventory,
        lead_time_demand=lead_time_demand,
        safety_stock_buffer=safety_stock_buffer
    )

    # 3. Replenishment Quantity Calculation Step
    suggested_replenishment = compute_replenishment_quantity(
        stockout_status=stockout_status,
        current_inventory=current_inventory,
        lead_time_demand=lead_time_demand,
        safety_stock_buffer=safety_stock_buffer
    )

    if stockout_status == "HIGH":
        risk_reason = (
            f"Expected demand during lead time ({lead_time_demand} units) reduces projected remaining "
            f"inventory to {remaining_inventory} units, below the safety buffer ({safety_stock_buffer} units)."
        )
    else:
        risk_reason = (
            f"Available inventory ({current_inventory} units) comfortably covers lead-time demand "
            f"({lead_time_demand} units) with {remaining_inventory} units remaining (>= {safety_stock_buffer} safety buffer)."
        )

    # 4. Estimated inventory runway (days)
    if avg_daily_demand > 0:
        runway_days = round(current_inventory / avg_daily_demand, 1)
    else:
        runway_days = 999.0

    return {
        "stock_code": stock_code,
        "description": desc,
        "available_inventory": current_inventory,
        "supplier_lead_time_days": supplier_lead_time,
        "safety_threshold_pct": round(safety_stock_threshold_pct * 100, 1),
        "safety_stock_buffer": safety_stock_buffer,
        "expected_lead_time_demand": lead_time_demand,
        "average_daily_demand": avg_daily_demand,
        "inventory_remaining_after_lead_time": remaining_inventory,
        "stockout_status": stockout_status,
        "risk_color": risk_color,
        "suggested_replenishment": suggested_replenishment,
        "estimated_runway_days": runway_days,
        "risk_analysis_summary": risk_reason,
        "assumptions": (
            "Replenishment restores on-hand inventory to cover D_lead plus safety buffer SS "
            "assuming stationary unit prices, immediate order dispatch, and user-supplied starting stock."
        ),
        "disclaimer": (
            "NOTICE: The UCI Online Retail dataset does not record warehouse physical inventory. "
            "This metric is an explicit forecast-driven inventory simulation based on ML lead-time demand."
        ),
        "daily_forecast_schedule": forecast_result['forecast_schedule']
    }
