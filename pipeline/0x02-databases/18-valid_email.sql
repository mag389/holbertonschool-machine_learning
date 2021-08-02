-- trigger when email changed
DELIMITER //

CREATE TRIGGER update_email
BEFORE UPDATE
ON users
FOR EACH ROW
	IF (NEW.email <> OLD.email) THEN
		SET NEW.valid_email = 0;
	END IF//
	-- UPDATE users
	-- SET NEW.valid_email = 0
	-- WHERE (NEW.email <> OLD.email)
DELIMITER ;
