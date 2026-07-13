CREATE DATABASE customer_churn;

USE customer_churn;

-- BASIC ANALYSIS
-- Q1. How many customers does the company have?

-- Business Reason

-- Management needs to know the size of the customer base.



-- There are total 2800 customers  


-- Q2. How many customers are in each segment?

-- Reason

-- Shows customer distribution.

SELECT Cluster_Label, COUNT(*) AS TOTAL_CUSTOMER
FROM customer_analytics_final
GROUP BY  Cluster_Label
ORDER BY TOTAL_CUSTOMER DESC;

-- Q3. What percentage of customers belong to each segment?

-- Reason

-- Helps understand customer composition.

SELECT Cluster_Label, COUNT(*) AS TOTAL_CUSTOMER,
ROUND(COUNT(*)*100.0 / (SELECT COUNT(*) FROM customer_analytics_final),2) AS PERCENTAGE
FROM customer_analytics_final
GROUP BY Cluster_Label;



-- Q4. Which plan is most popular?

-- Reason

-- Identify customer preferences.


SELECT plan_type, COUNT(*) AS TOTAL_CUSTOMER
FROM customer_analytics_final
GROUP BY  plan_type
ORDER BY TOTAL_CUSTOMER DESC;

-- Every plan almost have equal users but if  we check which  plan type has more loyal customer we get the real picture 

SELECT plan_type, Cluster_Label,  COUNT(*) AS TOTAL_CUSTOMER
FROM customer_analytics_final
GROUP BY  plan_type,Cluster_Label;



-- Q5. Average monthly fee by plan

-- Reason

-- Understand pricing.


SELECT plan_type,AVG(monthly_fee) AVERAGE_MONTHLIY_FEE
FROM customer_analytics_final
GROUP BY  plan_type
ORDER BY AVERAGE_MONTHLIY_FEE DESC;


-- Q6. Overall churn rate

-- Reason

-- The most important KPI.

SELECT ROUND(AVG(churn)*100,2) AS Overall_churn_rate
FROM customer_analytics_final;

-- Q7. Churn rate by plan

-- Reason

-- Which plans lose customers?

SELECT plan_type, 
ROUND(AVG(churn)*100,2)
FROM customer_analytics_final
GROUP BY plan_type;


-- Q8. Churn by customer segment

-- Reason

-- Which segment is most at risk?

SELECT Cluster_Label, 
ROUND(AVG(churn)*100,2) AS churn_rate
FROM customer_analytics_final
GROUP BY Cluster_Label;

-- Q9. Which customers have the highest churn probability?

-- Reason

-- Target retention campaigns.

SELECT user_id ,
signup_date, 
plan_type,
avg_weekly_usage_hours,
payment_failures,
last_login_days_ago,
churn_probability

FROM customer_analytics_final
ORDER BY churn_probability DESC
LIMIT 20;

-- Q10. High-risk customers (>80%)

-- Reason

-- Immediate action list.

SELECT *
FROM customer_analytics_final
WHERE churn_probability>=0.80;

-- Q11. Total monthly revenue

-- Reason

-- Executive KPI.

SELECT SUM(monthly_fee) AS total_monthly_fee
FROM customer_analytics_final;


-- Q12. Revenue by plan

-- Reason

-- Which plans generate the most money?

SELECT plan_type , SUM(monthly_fee) AS total_monthly_fee
FROM customer_analytics_final
GROUP BY plan_type;


-- Q13. Revenue by segment

-- Reason

-- Which customer groups are valuable?

SELECT
Cluster_Label,
SUM(monthly_fee) revenue
FROM customer_analytics_final
GROUP BY Cluster_Label;

-- Q14. Revenue at risk

-- Reason

-- Money likely to be lost due to churn.

SELECT  SUM(monthly_fee)  AS revenue_at_risk
FROM customer_analytics_final
WHERE churn_probability>0.80;

-- CUSTOMER BEHAVIOUR

-- Q15. Average tenure

SELECT ROUND(AVG(tenure_months),2) AS avg_tenure
FROM customer_analytics_final;


-- Q16. Average support tickets


SELECT
ROUND(AVG(support_tickets),2)
FROM customer_analytics_final;


-- Q17. Customers with many support tickets

-- Reason

-- Support issues often increase churn.

SELECT *
FROM customer_analytics_final
WHERE support_tickets>=5;


-- Q18. Customers inactive for more than 30 days

SELECT *
FROM customer_analytics_final
WHERE last_login_days_ago>30;


-- there numbers
with inactive as(SELECT *
FROM customer_analytics_final
WHERE last_login_days_ago>30)

SELECT Cluster_Label,COUNT(*) AS inactive_more_than_30_days FROM inactive 
GROUP BY Cluster_Label;



-- Q19. Customers with payment failures

SELECT *
FROM customer_analytics_final
WHERE payment_failures>0;

-- there numbers

WITH FALIURE as(SELECT *
FROM customer_analytics_final
WHERE payment_failures>0)

SELECT Cluster_Label,COUNT(*) AS payment_failiure FROM  FALIURE
GROUP BY Cluster_Label;


-- Q20. Average usage by segment

SELECT Cluster_Label, ROUND(AVG(avg_weekly_usage_hours),2) AS avg_usage
FROM customer_analytics_final
GROUP BY Cluster_Label;

-- Q21. Average Monthly Fee Within Each Customer Segment

SELECT Cluster_Label, ROUND(AVG(monthly_fee),2) AS avg_fee
FROM customer_analytics_final
GROUP BY Cluster_Label ;

-- 22. Which customer segment raises the most support tickets?
-- Business Question

-- Which customer segment requires the most customer support?

SELECT
    Cluster_Label,
    AVG(support_tickets) AS avg_tickets
FROM customer_analytics_final
GROUP BY Cluster_Label
ORDER BY avg_tickets DESC;

-- 23. Does customer inactivity increase churn?
-- Business Question

-- How does the number of days since the last login affect customer churn?

SELECT CASE
WHEN last_login_days_ago <=7 THEN "Active"
WHEN last_login_days_ago <30 THEN "Moderately Active"
ELSE  "Inactive"
END AS Ativity_level,
COUNT(*) customers,

ROUND(AVG(churn)*100,2) churn_rate

FROM customer_analytics_final
GROUP BY Ativity_level;


-- 24. Which tenure group has the highest churn? 
-- Business Question

-- Do new customers churn more than long-term customers?

SELECT
CASE
WHEN tenure_months<6 THEN 'New'
WHEN tenure_months BETWEEN 6 AND 12 THEN 'Growing'
ELSE 'Loyal'
END tenure_group,
COUNT(*) customers,
ROUND(AVG(churn)*100,2) churn_rate
FROM customer_analytics_final
GROUP BY tenure_group;

-- 25. Which subscription plan generates the highest revenue but also experiences high churn?
-- Business Question

-- Is the company's highest-earning subscription plan also losing the most customers?


SELECT plan_type,
SUM(monthly_fee)AS revenue,
ROUND(AVG(churn_probability)*100,2)AS churn_rate
FROM customer_analytics_final
GROUP BY plan_type




























