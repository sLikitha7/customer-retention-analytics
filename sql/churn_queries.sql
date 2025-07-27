
-- Silent Churn Detection SQL Queries
-- For use with PostgreSQL, MySQL, or similar databases

-- 1. Identify At-Risk Customers
SELECT 
    customer_id,
    segment,
    frustration_score,
    mrr,
    tenure_months,
    support_tickets_3m,
    days_since_last_login,
    customer_status
FROM customers 
WHERE customer_status = 'At Risk'
    AND mrr > 500  -- Focus on valuable customers
ORDER BY frustration_score DESC, mrr DESC
LIMIT 100;

-- 2. Customer Churn Risk Score Calculation
SELECT 
    customer_id,
    segment,
    -- Churn risk calculation based on multiple factors
    CASE 
        WHEN frustration_score > 70 OR days_since_last_login > 30 THEN 'High'
        WHEN frustration_score > 40 OR days_since_last_login > 14 THEN 'Medium'
        ELSE 'Low'
    END as churn_risk,
    
    -- Revenue at risk
    mrr as revenue_at_risk,
    
    -- Key risk indicators
    frustration_score,
    days_since_last_login,
    support_tickets_3m,
    avg_resolution_time_hours
    
FROM customers
ORDER BY frustration_score DESC;

-- 3. Segment-Based Churn Analysis
SELECT 
    segment,
    COUNT(*) as total_customers,
    COUNT(CASE WHEN customer_status = 'At Risk' THEN 1 END) as at_risk_customers,
    ROUND(
        COUNT(CASE WHEN customer_status = 'At Risk' THEN 1 END) * 100.0 / COUNT(*), 
        2
    ) as at_risk_percentage,
    
    AVG(frustration_score) as avg_frustration_score,
    AVG(mrr) as avg_mrr,
    SUM(CASE WHEN customer_status = 'At Risk' THEN mrr ELSE 0 END) as at_risk_mrr
    
FROM customers
GROUP BY segment
ORDER BY at_risk_percentage DESC;

-- 4. Support Ticket Trend Analysis
SELECT 
    segment,
    -- Support metrics by segment
    AVG(support_tickets_3m) as avg_tickets_per_customer,
    AVG(avg_resolution_time_hours) as avg_resolution_time,
    
    -- Correlation with frustration
    AVG(CASE WHEN support_tickets_3m > 3 THEN frustration_score END) as high_volume_frustration,
    AVG(CASE WHEN support_tickets_3m <= 1 THEN frustration_score END) as low_volume_frustration,
    
    -- At-risk customers with high support volume
    COUNT(CASE WHEN support_tickets_3m > 3 AND customer_status = 'At Risk' THEN 1 END) as high_volume_at_risk
    
FROM customers
GROUP BY segment
ORDER BY avg_tickets_per_customer DESC;

-- 5. Customer Health Scorecard
SELECT 
    customer_id,
    segment,
    
    -- Health indicators
    CASE 
        WHEN frustration_score < 30 THEN 'Excellent'
        WHEN frustration_score < 50 THEN 'Good'
        WHEN frustration_score < 70 THEN 'Fair'
        ELSE 'Poor'
    END as health_grade,
    
    -- Engagement score
    CASE 
        WHEN days_since_last_login <= 7 AND feature_adoption_score > 70 THEN 'Highly Engaged'
        WHEN days_since_last_login <= 14 AND feature_adoption_score > 40 THEN 'Engaged'
        WHEN days_since_last_login <= 30 THEN 'Moderately Engaged'
        ELSE 'Disengaged'
    END as engagement_level,
    
    -- Value tier
    CASE 
        WHEN mrr >= 1000 THEN 'Enterprise'
        WHEN mrr >= 200 THEN 'Growth'
        WHEN mrr >= 50 THEN 'Standard'
        ELSE 'Basic'
    END as value_tier,
    
    frustration_score,
    mrr,
    tenure_months
    
FROM customers
ORDER BY 
    CASE health_grade 
        WHEN 'Poor' THEN 1 
        WHEN 'Fair' THEN 2 
        WHEN 'Good' THEN 3 
        ELSE 4 
    END,
    mrr DESC;

-- 6. Monthly Churn Risk Trend (if you have historical data)
-- This would require a time-series table
/*
SELECT 
    DATE_TRUNC('month', analysis_date) as month,
    segment,
    
    AVG(frustration_score) as avg_frustration,
    COUNT(CASE WHEN customer_status = 'At Risk' THEN 1 END) as at_risk_count,
    COUNT(*) as total_customers,
    
    -- Trend calculation
    LAG(AVG(frustration_score)) OVER (
        PARTITION BY segment 
        ORDER BY DATE_TRUNC('month', analysis_date)
    ) as prev_month_frustration
    
FROM customer_history
WHERE analysis_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', analysis_date), segment
ORDER BY month DESC, segment;
*/

-- 7. High-Impact Intervention Candidates
SELECT 
    customer_id,
    segment,
    mrr,
    frustration_score,
    tenure_months,
    
    -- Intervention priority score
    (
        LEAST(mrr / 100, 50) +  -- Revenue impact (max 50 points)
        frustration_score * 0.3 +  -- Frustration level (max 30 points)  
        LEAST(tenure_months * 0.5, 20)  -- Tenure value (max 20 points)
    ) as intervention_priority,
    
    -- Specific issues
    CASE WHEN support_tickets_3m > 5 THEN 'High Support Volume, ' ELSE '' END ||
    CASE WHEN days_since_last_login > 14 THEN 'Low Engagement, ' ELSE '' END ||
    CASE WHEN feature_adoption_score < 30 THEN 'Poor Adoption, ' ELSE '' END ||
    CASE WHEN late_payments_6m > 0 THEN 'Payment Issues' ELSE '' END as key_issues
    
FROM customers
WHERE customer_status IN ('At Risk', 'Frustrated')
    AND mrr > 100  -- Focus on customers with meaningful revenue
ORDER BY intervention_priority DESC
LIMIT 50;
