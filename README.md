# Delivery Time Estimation Project

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

This project develops a machine learning solution to predict delivery times for a leading intra-city logistics company in India, operating within the $40 billion intra-city logistics sector. The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology and utilizes a dataset of **172,000 rows** (`datadelevry.xlsx`) to build and deploy predictive models.

## Project Overview

The client, a premier marketplace for intra-city logistics in India, supports over 150,000 driver-partners by ensuring consistent earnings and fostering independence. With a customer base exceeding 5 million, the company collaborates with various restaurants to facilitate direct delivery of goods. The primary goal is to provide accurate delivery time estimations based on order details, location, and available delivery partners.

### Objectives
- **Regression**: Predict delivery time in minutes using machine learning models.
- **Clustering**: Classify orders into meaningful groups for operational insights.

## Data Description

The dataset (`datadelevry.xlsx`) contains **172,000 rows**, each representing a unique delivery with the following features:

| Column Name                | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `market_id`                | Integer ID of the market where the restaurant is located                    |
| `created_at`               | Timestamp when the order was placed                                         |
| `actual_delivery_time`     | Timestamp when the order was delivered                                      |
| `store_primary_category`   | Category of the restaurant (e.g., Italian, Indian)                          |
| `order_protocol`           | Integer code for order protocol (e.g., porter, call, pre-booked, third-party) |
| `total_items_subtotal`     | Final price of the order                                                   |
| `num_distinct_items`       | Number of distinct items in the order                                      |
| `min_item_price`           | Price of the cheapest item in the order                                    |
| `max_item_price`           | Price of the costliest item in the order                                   |
| `total_onshift_partners`   | Number of delivery partners on duty at the time of order placement          |
| `total_busy_partners`      | Number of delivery partners attending other tasks                           |
| `total_outstanding_orders` | Total number of orders pending fulfillment at the time of order placement   |

## Methodology

The project adheres to the **CRISP-DM** methodology, with the following phases:
1. **Business Understanding**: Define objectives and deliverables to meet the client's need for accurate delivery time predictions.
2. **Data Understanding**: Explore and analyze the dataset to identify patterns and relationships.
3. **Data Preparation**: Clean, preprocess, and transform data (e.g., feature engineering, handling missing values).
4. **Modeling**: Develop regression models (e.g., LightGBM) for delivery time prediction and clustering models (e.g., DBSCAN) for order classification.
5. **Evaluation**: Assess model performance using metrics such as RMSE and MAE for regression, and silhouette score for clustering.
6. **Deployment**: Integrate the predictive model into production systems with actionable recommendations.

## Web Application

The project includes a **Flask-based web application** for real-time delivery time predictions and order clustering, leveraging two machine learning models:
- **LightGBM**: Predicts delivery time in minutes.
- **DBSCAN**: Clusters orders based on features for operational insights.

### Features
- **User Interface**: Responsive web interface built with Bootstrap 5, featuring an intuitive form for inputting order details and real-time prediction results.
- **Predictions**:
  - Delivery time in minutes (LightGBM).
  - Cluster classification with interpretable labels (DBSCAN).
- **Data Validation**: Robust client-side and server-side validation to ensure reliable input handling.
- **API Endpoints**:
  - `GET /`: Main page with the input form.
  - `POST /predict`: Returns predictions in JSON format.
  - `GET /health`: Checks application and model status.

### Input Features
The application accepts the following 13 input variables:

| Variable                   | Type   | Description                              |
|----------------------------|--------|------------------------------------------|
| `market_id`                | int    | Market identifier                        |
| `store_id`                 | int    | Store identifier                         |
| `store_primary_category`   | str    | Store category (e.g., Restaurant)        |
| `total_items`              | int    | Total number of items                    |
| `subtotal`                 | float  | Order subtotal                           |
| `num_distinct_items`       | int    | Number of distinct items                 |
| `total_onshift_partners`   | int    | Number of available delivery partners     |
| `delivery_duration_min`    | float  | Estimated delivery duration               |
| `day_of_week_numeric`      | int    | Day of the week (1–7)                    |
| `hour_of_day`              | int    | Hour of the day (0–23)                   |
| `price_range`              | str    | Price range (e.g., Budget, Mid-range)    |
| `weather_condition`        | str    | Weather condition (e.g., Clear, Cloudy)  |
| `temperature`              | float  | Temperature in °C                        |

## Project Structure

```
delivery-prediction-app/
├── app.py                    # Main Flask application
├── run.py                    # Startup script with validation checks
├── test_app.py               # Automated tests
├── templates/
│   └── index.html            # Responsive web interface
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variable template
├── nginx.conf               # Nginx configuration (optional)
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Docker Compose for orchestration
├── deploy.sh                 # Automated deployment script
├── README.md                # Project documentation
├── QUICK_START.md           # Quick start guide
├── PROJECT_STRUCTURE.md     # Project structure overview
├── sample_data.json         # Sample data for testing
├── lightgbm_model.pkl       # LightGBM model (to be added)
└── dbscan_model.pkl         # DBSCAN model (to be added)
```

## Requirements

- **Python**: 3.8 or higher
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `flask`, `gunicorn` (for production)
- **Optional**: `matplotlib`, `seaborn` (for visualization), Jupyter Notebook (for data exploration)
- **Docker**: For containerized deployment

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chamseddinedoulaEsprit/CRISP-DM-DataSet-Temps-de-livraison.git
   cd CRISP-DM-DataSet-Temps-de-livraison
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Machine Learning Models**:
   - Place `lightgbm_model.pkl` and `dbscan_model.pkl` in the project root directory.

4. **Configure Environment (Optional)**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to set `HOST`, `PORT`, and `DEBUG` as needed.

## Running the Application

### Option 1: Local Development
```bash
python run.py
```

### Option 2: Flask Direct
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

### Option 3: Docker
```bash
docker-compose up -d
```

Access the application at:
- **Local**: `http://localhost:5000`
- **Network**: `http://<your-ip>:5000`

## Deployment

### Production Deployment
1. **With Gunicorn**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **With Docker**:
   ```bash
   docker-compose up -d
   ```

3. **Environment Variables**:
   ```bash
   export FLASK_ENV=production
   export DEBUG=False
   ```

### Security Considerations
- Enable HTTPS in production.
- Disable debug mode (`DEBUG=False`).
- Validate all input data to prevent injection attacks.
- Use structured logging for monitoring.

## Customization

1. **Preprocessing**:
   - Modify `preprocess_features()` in `app.py` to align with the original data pipeline (e.g., encoding, scaling).
   ```python
   def preprocess_features(df):
       # Adapt according to your preprocessing pipeline
       # Encoding, normalization, feature engineering, etc.
       return df_processed
   ```

2. **Categories**:
   - Update lists in `app.py`:
     ```python
     STORE_CATEGORIES = ['Restaurant', 'Grocery', ...]
     WEATHER_CONDITIONS = ['Clear', 'Cloudy', ...]
     PRICE_RANGES = ['Budget', 'Mid-range', ...]
     ```

3. **Cluster Interpretation**:
   - Adjust `interpret_cluster()` in `app.py`:
     ```python
     def interpret_cluster(cluster_id):
         interpretations = {
             -1: "Atypical Order",
             0: "Standard Order",
         }
         return interpretations.get(cluster_id, f"Cluster {cluster_id}")
     ```

## API Endpoints

1. **GET /**:
   - Renders the main page with the input form.

2. **POST /predict**:
   - **Request**:
     ```json
     {
       "market_id": 1,
       "store_id": 100,
       "store_primary_category": "Restaurant",
       "total_items": 3,
       "subtotal": 25.50,
       "num_distinct_items": 2,
       "total_onshift_partners": 5,
       "delivery_duration_min": 30.0,
       "day_of_week_numeric": 3,
       "hour_of_day": 19,
       "price_range": "Mid-range",
       "weather_condition": "Clear",
       "temperature": 22.5
     }
     ```
   - **Response**:
     ```json
     {
       "success": true,
       "predictions": {
         "delivery_time_minutes": 25.5,
         "delivery_time_formatted": "25min",
         "cluster": 0,
         "cluster_interpretation": "Standard Order"
       }
     }
     ```

3. **GET /health**:
   - Returns application status and model availability.

## Testing

Run automated tests to validate functionality:
```bash
python test_app.py
```

For a specific port:
```bash
python test_app.py 8000
```

Check application health:
```bash
curl http://localhost:5000/health
```

## Troubleshooting

| Issue                     | Solution                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| Models not found          | Ensure `lightgbm_model.pkl` and `dbscan_model.pkl` are in the project root. |
| Prediction errors         | Verify `preprocess_features()` matches the training pipeline.             |
| Port already in use       | Update `PORT` in `.env` (e.g., `PORT=5001`).                             |
| Docker errors             | Check logs with `docker-compose logs`.                                   |

## Notes

- Ensure model files are compatible with the installed library versions.
- Adapt preprocessing to match the training pipeline for accurate predictions.
- Use caching for high-volume predictions to improve performance.
- Monitor the `/health` endpoint for application status in production.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

For technical issues or enhancements, refer to code comments in `app.py` or open an issue on the [GitHub repository](https://github.com/chamseddinedoulaEsprit/CRISP-DM-DataSet-Temps-de-livraison).
