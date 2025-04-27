CREATE DATABASE IF NOT EXISTS flights_db;
USE flights_db;
CREATE TABLE flights (
    year INT,
    month INT,
    carrier VARCHAR(5),
    carrier_name VARCHAR(100),
    airport VARCHAR(5),
    airport_name VARCHAR(100),
    arr_flights DECIMAL(10,2) NULL,
    arr_del15 DECIMAL(10,2) NULL,
    carrier_ct DECIMAL(10,2) NULL,
    weather_ct DECIMAL(10,2) NULL,
    nas_ct DECIMAL(10,2) NULL,
    security_ct DECIMAL(10,2) NULL,
    late_aircraft_ct DECIMAL(10,2) NULL,
    arr_cancelled DECIMAL(10,2) NULL,
    arr_diverted DECIMAL(10,2) NULL,
    arr_delay DECIMAL(10,2) NULL,
    carrier_delay DECIMAL(10,2) NULL,
    weather_delay DECIMAL(10,2) NULL,
    nas_delay DECIMAL(10,2) NULL,
    security_delay DECIMAL(10,2) NULL,
    late_aircraft_delay DECIMAL(10,2) NULL,
    PRIMARY KEY (year, month, carrier, airport)
);

-- Create indexes for better query performance
CREATE INDEX idx_carrier ON flights(carrier);
CREATE INDEX idx_airport ON flights(airport);
CREATE INDEX idx_year_month ON flights(year, month);

-- Check db after import
select* from flights;

-- Basic Delay Overview by Airline
SELECT 
    carrier_name AS airline,
    ROUND(AVG(arr_delay), 1) AS avg_delay_minutes,
    COUNT(*) AS total_flights
FROM flights
GROUP BY airline
ORDER BY avg_delay_minutes DESC;

-- Average Arrival Delay per Airline 
SELECT 
    carrier_name AS airline,
    ROUND(AVG(arr_delay), 2) AS avg_arrival_delay_minutes,
    COUNT(*) AS total_flights
FROM flights
WHERE arr_cancelled = 0  -- Only include non-cancelled flights
GROUP BY carrier_name
ORDER BY avg_arrival_delay_minutes DESC;


-- Total Flights per Airport (Busiest Airports)
SELECT 
    airport_name,
    SUM(arr_flights) AS total_flights
FROM flights
GROUP BY airport_name
ORDER BY total_flights DESC
LIMIT 10;

-- Busiest Airports by Total Flights
SELECT 
    airport_name,
    SUM(arr_flights) AS total_flights,
    ROUND(SUM(arr_del15) / SUM(arr_flights) * 100, 2) AS delay_percentage
FROM flights
GROUP BY airport_name
ORDER BY total_flights DESC
LIMIT 20;

-- Cancellation Analysis
SELECT 
    carrier_name,
    SUM(arr_cancelled) AS cancelled_flights,
    ROUND(SUM(arr_cancelled)/SUM(arr_flights)*100, 2) AS cancellation_rate
FROM flights
GROUP BY carrier_name
ORDER BY cancelled_flights DESC;

# Cancellation Reasons Breakdown
SELECT 
    CASE 
        WHEN carrier_ct > weather_ct AND carrier_ct > nas_ct AND carrier_ct > security_ct THEN 'Carrier'
        WHEN weather_ct > carrier_ct AND weather_ct > nas_ct AND weather_ct > security_ct THEN 'Weather'
        WHEN nas_ct > carrier_ct AND nas_ct > weather_ct AND nas_ct > security_ct THEN 'National Air System'
        WHEN security_ct > carrier_ct AND security_ct > weather_ct AND security_ct > nas_ct THEN 'Security'
        ELSE 'Other'
    END AS primary_cancellation_reason,
    COUNT(*) AS cancellation_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM flights WHERE arr_cancelled > 0), 2) AS percentage
FROM flights
WHERE arr_cancelled > 0
GROUP BY primary_cancellation_reason
ORDER BY cancellation_count DESC;

-- Delay Rate by Airport
SELECT 
    airport_name,
    ROUND(SUM(arr_del15)/SUM(arr_flights)*100, 1) AS delay_percentage
FROM flights
GROUP BY airport_name
HAVING SUM(arr_flights) > 100  -- Only airports with significant traffic
ORDER BY delay_percentage DESC
LIMIT 30;

# Monthly Delay Trends
SELECT 
    month,
    ROUND(AVG(arr_delay), 2) AS avg_delay_minutes,
    SUM(arr_flights) AS total_flights,
    ROUND(SUM(arr_del15) / SUM(arr_flights) * 100, 2) AS delay_percentage
FROM flights
GROUP BY month
ORDER BY month;

# Worst Days for Flight Delays
SELECT 
    CONCAT(year, '-', LPAD(month, 2, '0'), '-01') AS month_year,
    ROUND(AVG(arr_delay), 2) AS avg_delay_minutes,
    SUM(arr_flights) AS total_flights
FROM flights
GROUP BY year, month
ORDER BY avg_delay_minutes DESC
LIMIT 10;

# Airlines with Highest Late Aircraft Delays
SELECT 
    carrier_name,
    ROUND(SUM(late_aircraft_delay) / SUM(arr_delay) * 100, 2) AS late_aircraft_delay_percentage,
    ROUND(AVG(late_aircraft_delay), 2) AS avg_late_aircraft_delay_minutes
FROM flights
WHERE arr_delay > 0
GROUP BY carrier_name
ORDER BY late_aircraft_delay_percentage DESC
LIMIT 10;

-- Airlines with Best On-Time Performance
SELECT 
    carrier_name,
    ROUND(SUM(arr_del15)/SUM(arr_flights)*100, 1) AS delay_percentage
FROM flights
GROUP BY carrier_name
ORDER BY delay_percentage ASC
LIMIT 5;

# Weather Impact Analysis
SELECT 
    airport_name,
    SUM(weather_ct) AS weather_delays_count,
    ROUND(SUM(weather_delay) / NULLIF(SUM(weather_ct), 0), 2) AS avg_weather_delay_minutes
FROM flights
WHERE weather_ct > 0
GROUP BY airport_name
ORDER BY weather_delays_count DESC
LIMIT 20;

