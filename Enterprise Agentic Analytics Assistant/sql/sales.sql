CREATE TABLE sales (
    sale_id INT PRIMARY KEY,
    customer_id INT,
    region_id INT,
    sale_date DATE,
    amount DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);