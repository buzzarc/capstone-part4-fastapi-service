# Model Monitoring & Responsible-Use Plan

## 1. Post-Deployment Monitoring
* **Data Drift Tracking:** Population Stability Index (PSI) values will be calculated weekly on critical features like `recency` and `support_tickets`. A $PSI > 0.25$ triggers automated email notifications for input shift markers.
* **Prediction Volatility:** Track daily statistical quantiles of emitted churn probabilities to catch upstream tracking data errors or sudden system deviations.
* **API Reliability Metrics:** Track API latency metrics (Target: < 50ms at p95 boundaries) and monitor HTTP 5xx error statuses dynamically.

## 2. Retraining Framework
Retraining runs will be triggered automatically if:
* The actual observed churn rate metrics deviate from predicted probability distribution curves by more than 15%.
* Data drift alerts remain continuously active for more than 14 consecutive calendar days.
* A standard 90-day production loop passes, ensuring the network updates with fresh user trends.

## 3. Responsible Use Plan
* **Approved Use Cases:** Ordering priority outreach lines for customer service agents, routing struggling high-value profiles to specialized teams, and engineering user interface components for highly active categories.
* **Prohibited Use Cases:** Using risk scoring patterns to apply dynamic markups on product prices, or rejecting standard product refund options based entirely on a high predicted churn metric.