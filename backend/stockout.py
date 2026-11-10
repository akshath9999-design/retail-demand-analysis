"""
backend/stockout.py
Stockout Risk Analysis and Inventory Optimization Engine.
Applies the explicit forecast-driven inventory rules and safety thresholds.
"""

from typing import Dict, Any, Optional
from backend.prediction import forecaster

def analyze_stockout_risk(
    stock_code: str,
    current_inventory: int,
    supplier_lead_time: int,
    safety_stock_threshold_pct: float = 0.20
) -> Dict[str, Any]:
    """
    Evaluates stockout risk based on model-forecasted demand over the supplier lead time.

    Rule:
      Lead-Time Demand (D_lead) = Sum of predicted daily demand for 'supplier_lead_time' days.
      Safety Stock Buffer (SS) = Round(D_lead * safety_stock_threshold_pct).
      Remaining Inventory = current_inventory - D_lead.
      
      Stockout Risk = HIGH if D_lead >= current_inventory OR (current_inventory - D_lead) < SS.
      Otherwise LOW.

      Suggested Replenishment Quantity:
        If HIGH risk: Max(0, Round(D_lead + SS - current_inventory))
        If LOW risk: 0 (or optional maintenance reorder)
    """
    stock_code = stock_code.strip().upper()
    if supplier_lead_time < 1:
        raise ValueError("Supplier lead time must be at least 1 day.")
    if current_inventory < 0:
        raise ValueError("Available inventory cannot be negative.")

    # 1. Run model forecast for supplier_lead_time days
    forecast_result = forecaster.forecast(stock_code, forecast_days=supplier_lead_time)

    # 2. Extract metrics
    lead_time_demand = float(forecast_result['total_forecast_demand'])
    avg_daily_demand = float(forecast_result['avg_daily_demand'])
    desc = forecast_result['description']

    # 3. Calculate safety stock and remaining inventory
    safety_stock_buffer = round(lead_time_demand * safety_stock_threshold_pct)
    safety_stock_buffer = max(2, safety_stock_buffer) if lead_time_demand > 0 else 0
    
    remaining_inventory = round(current_inventory - lead_time_demand, 1)

    # 4. Explicit Inventory Rule:
    # Stockout Risk = HIGH when forecast demand during the lead time >= available inventory
    # Or if remaining inventory falls below the safety buffer
    is_deficit = lead_time_demand >= current_inventory
    is_below_safety = remaining_inventory < safety_stock_buffer

    if is_deficit or is_below_safety:
        stockout_status = "HIGH"
        risk_color = "#dc3545" # Crimson red
        # Replenishment recommendation to restore full inventory + safety buffer
        suggested_replenishment = max(0, round((lead_time_demand + safety_stock_buffer) - current_inventory))
        risk_reason = (
            f"Expected demand during lead time ({lead_time_demand} units) exceeds or critically breaches "
            f"current inventory ({current_inventory} units) including safety buffer ({safety_stock_buffer} units)."
        )
    else:
        stockout_status = "LOW"
        risk_color = "#28a745" # Forest green
        suggested_replenishment = 0
        risk_reason = (
            f"Available inventory ({current_inventory} units) comfortably covers lead-time demand "
            f"({lead_time_demand} units) with {remaining_inventory} units remaining above safety buffer."
        )

    # 5. Inventory runway (estimated days until stock depletion)
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
        "disclaimer": (
            "NOTICE: The Online Retail dataset does not record warehouse physical inventory. "
            "This metric is an explicit forecast-driven inventory simulation based on ML lead-time demand."
        ),
        "daily_forecast_schedule": forecast_result['forecast_schedule']
    }
