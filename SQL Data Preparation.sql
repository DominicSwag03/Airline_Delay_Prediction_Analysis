-- Create a view with cleaned data
CREATE VIEW flight_delay_analysis AS
SELECT 
    year,
    month,
    carrier,
    carrier_name,
    airport,
    airport_name,
    arr_flights,
    arr_del15,
    arr_delay,
    carrier_ct,
    weather_ct,
    nas_ct,
    security_ct,
    late_aircraft_ct,
    arr_cancelled,
    arr_diverted,
    -- Calculate delay percentages
    CASE WHEN arr_flights > 0 THEN arr_del15/arr_flights ELSE 0 END AS delay_percentage,
    -- Create delay categories
    CASE 
        WHEN arr_delay <= 15 THEN 'On-time'
        WHEN arr_delay <= 60 THEN 'Small delay'
        WHEN arr_delay <= 180 THEN 'Medium delay'
        ELSE 'Large delay'
    END AS delay_category
FROM flights
WHERE arr_cancelled = 0;