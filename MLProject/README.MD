## Advance – CI/CD MLflow & Docker (4 pts)

Pada tahap advance, dibuat workflow CI menggunakan GitHub Actions untuk:

1. Menjalankan training model menggunakan MLflow Project
2. Menyimpan artefak model (mlruns) ke repository GitHub
3. Melakukan build Docker image dari MLflow model menggunakan `mlflow models build-docker`
4. Push Docker image secara otomatis ke Docker Hub

Workflow akan berjalan otomatis setiap ada push ke branch `main`.

### Docker Image
Docker image hasil training tersedia di Docker Hub:
- Repository: `creditcard-mlflow`
- Dibangun otomatis melalui CI

### Tools
- MLflow
- GitHub Actions
- Docker & Docker Hub
