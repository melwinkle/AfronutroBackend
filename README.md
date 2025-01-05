# AfronutroBackend

## Overview

The AfronutroBackend is a comprehensive backend system designed to provide personalized meal plans based on user preferences, dietary restrictions, and health goals. The system uses a hybrid recommender approach, combining content-based and deep learning-based recommenders to generate accurate and personalized meal recommendations.

## Django Sub-Applications

The Django project consists of three sub-applications:

### Users

- **Purpose**: Manages user authentication, profiles, and preferences.
- **Endpoints**: Provides endpoints for user registration, login, profile management, and preference settings.

### Recipes

- **Purpose**: Manages the recipes database, including CRUD operations for recipes.
- **Endpoints**: Provides endpoints for adding, updating, deleting, and retrieving recipes.

### Meal Planner

- **Purpose**: Generates personalized meal plans based on user preferences, dietary restrictions, and health goals.
- **Endpoints**: Provides endpoints for generating and retrieving meal plans.

## Features

- **Content-Based Recommender**: Uses TF-IDF vectorization to recommend recipes based on their content.
- **Deep Learning Recommender**: Uses a deep learning model to recommend recipes based on user and recipe features.
- **Hybrid Recommender**: Combines the content-based and deep learning recommenders to provide more accurate recommendations.
- **Rule-Based Filtering**: Filters recipes based on user preferences, dietary restrictions, and health goals.
- **Meal Plans**: Generates meal plans that match the user's TDEE and ensures that the total calories for any combination of one recipe from each meal type do not exceed the user's TDEE.

## Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Node.js and npm](https://nodejs.org/)
- [PostgreSQL](https://www.postgresql.org/download/)
- [Redis](https://redis.io/download)

### Clone the Repository

```bash
git clone https://github.com/yourusername/AfronutroBackend.git
cd AfronutroBackend
```

### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Install Node.js Modules

```bash
npm install
```

## Docker Setup

### Build and Run the Services

To build and run the services using Docker Compose, use the following command:

```bash
docker-compose up --build
```

This command will build and start the following services:

- **redis**: Redis server for caching and message brokering.
- **db**: PostgreSQL database server.
- **django**: Django application server running the backend.
- **celery**: Celery worker for handling asynchronous tasks.
- **api-gateway**: API gateway for routing requests to the Django backend.
- **pgadmin**: pgAdmin for managing the PostgreSQL database.

### Stopping the Services

To stop the services, use the following command:

```bash
docker-compose down
```

## Why These Technologies?

### Django

Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It is used for the backend application server to handle HTTP requests, manage the database, and serve the API endpoints.

### Node.js and npm

Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine. npm is the package manager for Node.js. They are used for the frontend development and managing dependencies.

### Redis

Redis is an in-memory data structure store used as a database, cache, and message broker. It is used for caching and message brokering in the backend.

### Celery

Celery is an asynchronous task queue/job queue based on distributed message passing. It is used for handling background tasks and scheduling periodic tasks.

### Docker

Docker is a platform for developing, shipping, and running applications in containers. Docker Compose is used to define and manage multi-container Docker applications. Docker ensures that the application runs consistently across different environments and simplifies the deployment process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
