# Real Estate MLOps Capstone

This is a student-friendly local MLOps demo: data generation → training/evaluation → tests → BentoML API → Docker → Minikube/Kubernetes. Airflow orchestrates training. GitHub Actions tests and builds/pushes an image.

**Dataset honesty:** the included dataset is synthetic demo data, not actual Pune market listings.

## 1. Local setup (PowerShell terminal in this folder)
Install Python 3.11, Git, Docker Desktop and VS Code. Then:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_data
python -m src.train
pytest -q
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

## 2. Run BentoML API
```powershell
bentoml serve src.service:HousingService --host 127.0.0.1 --port 3000
```
In a second terminal:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3000/predict -ContentType "application/json" -Body '{"area_sqft":1200,"bedrooms":2,"bathrooms":2,"age_years":8,"location_score":7}'
```
Stop with Ctrl+C.

## 3. Airflow
Docker Desktop must be running. From project root:
```powershell
docker compose -f airflow/docker-compose.yaml up airflow-init
docker compose -f airflow/docker-compose.yaml up -d
```
Open http://localhost:8080 (demo login airflow / airflow), enable and trigger `real_estate_training_pipeline`. Stop with:
```powershell
docker compose -f airflow/docker-compose.yaml down
```
This Compose setup is for local coursework only, not production.

## 4. Docker image
Train first, then:
```powershell
docker build -t real-estate-predictor:v1 .
docker run --rm -p 3000:3000 real-estate-predictor:v1
```

## 5. Minikube
Install Minikube and kubectl, then:
```powershell
minikube start --driver=docker
docker build -t real-estate-predictor:v1 .
minikube image load real-estate-predictor:v1
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/real-estate-predictor
kubectl port-forward service/real-estate-predictor 3000:3000
```
Leave port-forward running and test the prediction request above. To demonstrate rollout, build/load a new tag and update the image tag in `k8s/deployment.yaml`, then apply and check rollout status.

## 6. Push to your GitHub
Create an empty repository named `real-estate-mlops`, then run:
```powershell
git init
git add .
git commit -m "Initial real estate MLOps capstone"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/real-estate-mlops.git
git push -u origin main
```
Replace YOUR_USERNAME with your GitHub username. Never paste access tokens into code.

GitHub Actions runs tests and builds/pushes to GHCR on main. It cannot directly deploy to your laptop's Minikube from a GitHub-hosted runner; deploy locally with the commands above unless you deliberately configure a secure self-hosted runner.

## What each folder does
- `src/generate_data.py`: reproducible synthetic data
- `src/train.py`: train, evaluate MAE/R², reject failed models
- `src/service.py`: BentoML prediction API
- `airflow/`: local Airflow Compose configuration
- `airflow/dags/`: training DAG
- `tests/`: quality tests
- `Dockerfile`: container image
- `k8s/`: Kubernetes deployment and service
- `.github/workflows/ci.yml`: CI + image publishing
