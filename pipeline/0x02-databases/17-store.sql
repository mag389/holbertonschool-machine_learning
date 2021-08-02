-- trigger to decrease quantity
CREATE TRIGGER decrease_quantity
AFTER INSERT
ON orders
FOR EACH ROW
	UPDATE items
	SET quantity = quantity - NEW.number
	WHERE items.name = NEW.item_name;
