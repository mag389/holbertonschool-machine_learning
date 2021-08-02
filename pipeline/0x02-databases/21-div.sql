-- division function
DELIMITER $$

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
	BEGIN
		IF b = 0 THEN
			RETURN 0;
		END IF;
		RETURN (a / b);
	END $$

DELIMITER ;
