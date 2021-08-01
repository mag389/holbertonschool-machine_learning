-- max temps by state
-- selects state and max temp from that state grouped by state
SELECT state, MAX(value) AS max_temp 
FROM temperatures 
GROUP BY state
ORDER BY state ASC;
