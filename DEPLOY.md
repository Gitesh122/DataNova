# 🚀 Deploying DataNova

*Designed & developed by Gitesh Kund*

The app also has a **Deploy** page (sidebar → Share → Deploy) with a readiness check, these steps,
a secrets template and a deploy-ready ZIP download.

## Option 1: Streamlit Community Cloud (free, recommended)

### 1. Put the project on GitHub
Install Git from <https://git-scm.com>, then in the VS Code terminal inside the project folder (where `app.py` is):

```bash
git init
git add .
git commit -m "DataNova first deploy"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/DataNova.git
git push -u origin main
```
(Create the empty repository first at <https://github.com/new>. Without Git, use *Add file → Upload files* on GitHub.)

### 2. Deploy
1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository `YOUR-USERNAME/DataNova`, branch `main`, main file path **`app.py`**, and pick an app URL.
4. **Advanced settings** → Python **3.12** → paste secrets (below) → **Deploy**.

The first build takes 3–5 minutes. Every later `git push` redeploys automatically.

### 3. Secrets (optional, recommended)
```toml
DATANOVA_ADMINS = "gitesh"          # who is admin on the public app
DATANOVA_REQUIRE_LOGIN = "1"        # "0" = skip login, everyone is a guest
# DATANOVA_MONGO_URI = "mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/"   # permanent accounts
# ANTHROPIC_API_KEY = "sk-ant-..."  # AI analyst (also add 'anthropic' to requirements.txt)
```

### Good to know
- Free apps sleep after a few days without visitors; opening the link wakes them up.
- About 1 GB RAM, so keep uploaded files reasonably small.
- Storage is temporary: SQLite accounts and share links reset on restart. Use MongoDB Atlas (free) for permanent accounts.

## Option 2: Docker (Render, Railway, Hugging Face Spaces, any server)
```bash
docker build -t datanova .
docker run -p 8501:8501 -e DATANOVA_ADMINS=gitesh datanova
```
On Render or Railway, connect the GitHub repo. The Dockerfile is detected automatically and `$PORT` is respected.

## Option 3: Your own PC / server on the network
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Others on the same network open `http://YOUR-PC-IP:8501`.

## Troubleshooting a deployment
| Symptom | Fix |
|---|---|
| "ModuleNotFoundError" in the cloud logs | The package is missing from `requirements.txt`; add it and push |
| App shows "needs Streamlit 1.50" | Reboot the app in Streamlit Cloud, which reinstalls the latest Streamlit |
| Login works but accounts disappear | Expected on free hosting; set `DATANOVA_MONGO_URI` |
| Share links point to localhost | Links use the current address automatically; edit it in *Reports → Share link* if needed |
