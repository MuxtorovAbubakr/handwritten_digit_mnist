# Handwritten Digit MNIST

A Streamlit drawing interface and a FastAPI service that classifies handwritten digits with a PyTorch model.

## Run with Docker Compose

Docker Engine and the Docker Compose plugin are required. The trained checkpoint is not stored in Git, so first place it at `models/mnist_model.pth`:

```text
models/
└── mnist_model.pth
```

If you cloned the repository without the checkpoint, copy or download the trained file from your own model-artifact location before starting the services. Compose mounts the checkpoint into the API container as a read-only file; it is not included in either image.

Build and start the app:

```sh
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501) to draw and classify a digit. The API is available to the Streamlit container on the Compose network and is not published directly to the host.

Stop the services with `Ctrl+C`, or run `docker compose down` in another terminal.

## Run without Docker

Install the API dependencies from `requirements-api.txt` and the UI dependencies from `requirements-streamlit.txt` in the Python environments for those services. Start the API with `uvicorn api.main:app --reload`, then start the UI with `streamlit run app/streamlit_app.py`. The UI defaults to `http://127.0.0.1:8000/predict`; set `API_URL` to override it.
