# Commands used
Start Workbench VM
Create new Firewall rule and attach rule to the VM
ssh into the VM


## start the mlflow server
screen -S mlflow_server

go to the same folder where train and inference scripts are going to be run
add IP address of VM and localhost to allowed-hosts in mlflow server

mlflow server \
  --host 0.0.0.0 \
  --port 8100 \
  --allowed-hosts 34.56.132.152,34.56.132.152:8100,localhost,localhost:8100,127.0.0.1,127.0.0.1:8100 \
  --cors-allowed-origins '*' \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --serve-artifacts

Ctrl + A, then D to detach from the screen
screen -R mlflow_server -> to reattach


## train
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

gsutil cp gs://mlops-week1-operating-edge-473204-j5/training_data/raw/iris.csv ./data/
python train.py


## inference
gsutil cp gs://mlops-week1-operating-edge-473204-j5/training_data/v2/iris_inference.csv ./data/
python inference.py
