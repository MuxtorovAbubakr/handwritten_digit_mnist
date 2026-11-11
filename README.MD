# Handwritten Digit Recognition

A full-stack machine learning application that recognizes handwritten digits (0–9) using a PyTorch neural network. Users can draw a digit through an interactive Streamlit interface and receive a prediction from a FastAPI backend.

The application separates the user interface and model inference into independent services and uses Docker Compose to run them together.

## Overview

The application allows users to:
- Draw a handwritten digit using an interactive drawing canvas.
- Preview the drawing as a 28 × 28 grayscale image.
- Send the image to a FastAPI prediction endpoint.
- Receive the model's predicted digit.
- Undo drawing strokes or clear the canvas.

## Architecture

The application consists of three main components:

| Component | Technology | Responsibility |
|---|---|---|
| Machine Learning Model | PyTorch | Classifies handwritten digits |
| Frontend | Streamlit | Provides the interactive drawing interface |
| Backend API | FastAPI | Receives image data and returns predictions |
| Containerization | Docker | Packages the application services |
| Service Orchestration | Docker Compose | Runs and connects the services |

### Prediction Workflow

1. The user draws a digit on the Streamlit canvas.
2. The drawing is represented as a 28 × 28 grayscale image.
3. The frontend converts pixel values into a normalized numerical array.
4. The image is sent to the FastAPI `/predict` endpoint.
5. The backend validates the input and applies MNIST normalization.
6. The PyTorch model performs inference.
7. The API returns the predicted digit.
8. Streamlit displays the prediction.

### System Architecture

```text
                  User
                   |
                   v
          +------------------+
          |    Streamlit     |
          |  Drawing Canvas  |
          +------------------+
                   |
                   | HTTP POST
                   v
          +------------------+
          |     FastAPI      |
          |  /predict        |
          +------------------+
                   |
                   v
          +------------------+
          |   PyTorch Model  |
          |                  |
          |   784 -> 128     |
          |       ReLU       |
          |   128 -> 10      |
          +------------------+
                   |
                   v
            Predicted Digit
                   |
                   v
          +------------------+
          |    Streamlit     |
          | Display Result   |
          +------------------+
```

## Model Architecture

The application uses a fully connected neural network implemented with PyTorch.

The model takes a 28 × 28 grayscale image as input, flattens it into a 784-dimensional vector, and passes it through two linear layers.

| Layer | Configuration |
|---|---|
| Input | 784 features |
| Hidden Layer | 128 neurons |
| Activation | ReLU |
| Output Layer | 10 neurons |

The output represents scores for the ten possible digit classes (0–9). The predicted class is selected using the highest output score.

## Tech Stack

- **Python** — Main programming language
- **PyTorch** — Neural network implementation and inference
- **NumPy** — Numerical operations and image preprocessing
- **Streamlit** — Interactive user interface
- **FastAPI** — Backend prediction API
- **Pydantic** — Request data validation
- **Docker** — Containerization
- **Docker Compose** — Multi-container orchestration

## Project Structure

```text
handwritten_digit_mnist/
│
├── api/
│   ├── main.py
│   └── model_definition.py
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   └── ...
│
├── models/
│   └── mnist_model.pth
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── compose.yaml
│
├── requirements-api.txt
├── requirements-streamlit.txt
│
├── .dockerignore
├── .gitignore
└── README.MD
```

**Note:** The trained model checkpoint is not included in the Git repository.

## Getting Started

### Prerequisites

To run the application using Docker, you need:

- Docker Engine
- Docker Compose plugin
- The trained model checkpoint (`mnist_model.pth`)

### 1. Clone the Repository

```bash
git clone https://github.com/MuxtorovAbubakr/handwritten_digit_mnist.git

cd handwritten_digit_mnist
```

### 2. Add the Trained Model

Place your trained model checkpoint in the following location:

```text
models/
└── mnist_model.pth
```

The checkpoint must contain the PyTorch model's state dictionary and match the architecture defined in `api/model_definition.py`.

The checkpoint is intentionally excluded from Git and is mounted into the API container as a read-only file.

### 3. Build and Start the Application

```bash
docker compose up --build
```

Docker Compose builds and starts both services. The Streamlit service waits for the API health check before starting.

### 4. Open the Application

Once the containers are running, open:

http://localhost:8501

Draw a digit and click **Classify** to receive a prediction.

### 5. Stop the Application

Press `Ctrl+C` or execute:

```bash
docker compose down
```

## Running Without Docker

You can also run the services independently.

### 1. Install Dependencies

Install the API dependencies:

```bash
pip install -r requirements-api.txt
```

Install the Streamlit dependencies in another environment or the same environment:

```bash
pip install -r requirements-streamlit.txt
```

### 2. Start the FastAPI Backend

From the project root:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### 3. Start the Streamlit Frontend

Open another terminal and execute:

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

By default, the frontend sends requests to:

```text
http://127.0.0.1:8000/predict
```

You can override the API endpoint using the `API_URL` environment variable.

## API

### Prediction Endpoint

**POST** `/predict`

Accepts a 28 × 28 grayscale image represented as a two-dimensional array of pixel values.

Example request:

```json
{
  "pixels": [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
  ]
}
```

*The example above is abbreviated. The actual endpoint requires exactly 28 rows, each containing 28 values.*

Example response:

```json
{
  "digit": 7
}
```

The API validates the input dimensions and pixel range before performing inference.

## Engineering Considerations

Some implementation decisions in this project include:

- **Service separation:** The frontend and backend run as independent services.
- **Model loading:** The trained model is loaded once during API startup rather than on every prediction request.
- **Input validation:** The API checks image dimensions, pixel ranges, and empty drawings.
- **Inference mode:** PyTorch inference runs using `torch.inference_mode()`.
- **Containerization:** Separate Dockerfiles are used for the API and Streamlit services.
- **Health checks:** Docker Compose checks API availability before starting the frontend.
- **Checkpoint management:** The trained model is mounted into the API container as a read-only file.

## Future Improvements

- Add model evaluation results and performance metrics.
- Improve prediction robustness for different handwriting styles.
- Add automated tests for the API and model inference.
- Introduce structured logging and better monitoring.
- Explore model optimization and inference latency improvements.
- Deploy the application to a cloud environment.

## Author

**Abubakr Muxtorov**

GitHub: [MuxtorovAbubakr](https://github.com/MuxtorovAbubakr)