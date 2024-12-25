# Excalibur_AI_project1

## Backend
```shell
# on macos you need python3.9
brew install python@3.9
cd app/backend
python3.9 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Start web server:
```
fastapi dev
```
## Frontend
open another console

```
cd app/frontend
python3.9 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
streamlit run main.py
```