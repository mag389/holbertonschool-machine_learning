-- avergae temperatures from a bigegr db
-- grouped by city order by temp edsc
SELECT city, AVG(value) AS `avg_temp` FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
