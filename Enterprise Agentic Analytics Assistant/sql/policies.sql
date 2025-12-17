CREATE TABLE policies (
    policy_id INT PRIMARY KEY,
    policy_type VARCHAR(100),
    customer_id INT,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);