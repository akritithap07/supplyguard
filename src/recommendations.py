from __future__ import annotations


def risk_band(probability: float) -> str:
    """Map probability to a transparent operational risk band."""
    p = float(probability)
    if p >= 0.80:
        return "Critical"
    if p >= 0.60:
        return "High"
    if p >= 0.30:
        return "Moderate"
    return "Low"


def recommend_actions(probability: float, row: dict) -> list[str]:
    """Transparent rules for converting risk signals into actions."""
    p = float(probability)
    actions: list[str] = []
    shipping = str(row.get("Shipping Mode", ""))
    priority = str(row.get("Order Priority", ""))

    if p >= 0.80:
        actions.append("Prioritize fulfillment and review the shipment for expedited handling.")
        actions.append("Consider proactive customer communication about delivery risk.")
    elif p >= 0.60:
        actions.append("Flag the shipment for operational monitoring and warehouse prioritization.")
    elif p >= 0.30:
        actions.append("Monitor the shipment; no immediate escalation is recommended.")
    else:
        actions.append("Standard monitoring is appropriate.")

    if shipping == "Standard Class" and p >= 0.60:
        actions.append("Evaluate whether a faster shipping mode is operationally feasible.")
    if priority in {"High", "Critical"} and p >= 0.60:
        actions.append("Escalate because customer/order priority is high and predicted risk is elevated.")

    return actions
