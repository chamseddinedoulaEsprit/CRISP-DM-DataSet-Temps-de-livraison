# Delivery Time Estimation Project

This project is focused on predicting delivery times for a leading intra-city logistics company in India. The project is implemented using the **CRISP-DM methodology** on a dataset of **172,000 rows** (`datadelevry.xlsx`).

---

## Project Overview

This delivery company (our client) is India's foremost marketplace for intra-city logistics, spearheading innovation in the nation's $40 billion intra-city logistics sector. With a mission to enhance the livelihoods of over **150,000 driver-partners**, our client ensures consistent earnings and fosters independence among its workforce. Currently, the company boasts a customer base exceeding **5 million**.

Collaborating with a diverse array of restaurants, the company facilitates the direct delivery of their goods to consumers. Leveraging a network of delivery partners sourced from various eateries, our client seeks to provide customers with **accurate estimated delivery times** based on factors such as their order, location, and available delivery partners.

---

## Associated Tasks

- **Regression**: Prediction of the delivery time estimation.

---

## Data Description

The dataset `datadelevry.xlsx` contains **172,000 rows**, each corresponding to a unique delivery. Each column represents a feature as described below:

| Column Name | Description |
|-------------|-------------|
| `market_id` | Integer ID for the market where the restaurant lies |
| `created_at` | Timestamp when the order was placed |
| `actual_delivery_time` | Timestamp when the order was delivered |
| `store_primary_category` | Category of the restaurant |
| `order_protocol` | Integer code for order protocol (e.g., via porter, call to restaurant, pre-booked, third party) |
| `total_items_subtotal` | Final price of the order |
| `num_distinct_items` | Number of distinct items in the order |
| `min_item_price` | Price of the cheapest item in the order |
| `max_item_price` | Price of the costliest item in the order |
| `total_onshift_partners` | Number of delivery partners on duty at the time the order was placed |
| `total_busy_partners` | Number of delivery partners attending other tasks |
| `total_outstanding_orders` | Total number of orders to be fulfilled at that moment |

---

## Methodology

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, which includes the following phases:

1. **Business Understanding** – Define project objectives and deliverables.
2. **Data Understanding** – Explore and describe the dataset.
3. **Data Preparation** – Clean, format, and transform data for analysis.
4. **Modeling** – Apply regression models to predict delivery times.
5. **Evaluation** – Assess model performance and validate results.
6. **Deployment** – Provide recommendations for integrating the predictive model.

---

## Usage

1. Load the dataset `datadelevry.xlsx` into your preferred Python environment (e.g., Pandas).
2. Explore and preprocess the data.
3. Train regression models to predict delivery times.
4. Evaluate model performance using appropriate metrics (e.g., RMSE, MAE).
5. Deploy the model to estimate delivery times for future orders.

---

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook (optional)

---



