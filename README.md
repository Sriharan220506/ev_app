# EV Battery Analytics - Backend with PostgreSQL

FastAPI backend for ESP32 telemetry with PostgreSQL database on Railway.

---

## 📋 STEP-BY-STEP DEPLOYMENT GUIDE

### Step 1: Create Railway Account

1. Go to **[railway.app](https://railway.app)**
2. Click **"Login"** → Sign in with **GitHub**
3. Verify your email if prompted

---

### Step 2: Create New Project

1. Click **"New Project"** (purple button)
2. Select **"Empty Project"**
3. Your project is created! You'll see an empty canvas.

---

### Step 3: Add PostgreSQL Database

1. Click **"+ Add Service"** (or right-click on canvas)
2. Select **"Database"** → **"Add PostgreSQL"**
3. Wait for PostgreSQL to deploy (green checkmark)
4. Click on the PostgreSQL service
5. Go to **"Variables"** tab
6. Copy the **`DATABASE_URL`** value (you'll need this later)

```
Example: postgresql://postgres:password@containers-xyz.railway.app:5432/railway
```

---

### Step 4: Deploy Backend Code

**Option A: Deploy from GitHub (Recommended)**

1. Push your `backend` folder to GitHub:
```bash
cd c:\Users\SriHaran\Documents\Projects\evapp\backend
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ev-battery-backend.git
git push -u origin main
```

2. In Railway project:
   - Click **"+ Add Service"** → **"GitHub Repo"**
   - Select your repository
   - Railway auto-detects Python and builds

**Option B: Deploy with Railway CLI**

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and deploy:
```bash
cd c:\Users\SriHaran\Documents\Projects\evapp\backend
railway login
railway link  # Select your project
railway up
```

---

### Step 5: Connect Backend to Database

Railway automatically connects services in the same project, but verify:

1. Click on your **Backend service**
2. Go to **"Variables"** tab
3. Click **"+ Add Variable Reference"**
4. Select **PostgreSQL** → **`DATABASE_URL`**
5. Click **"Add"**

Your backend now has access to the database!

---

### Step 6: Get Your Backend URL

1. Click on your **Backend service**
2. Go to **"Settings"** tab
3. Under **"Networking"** → **"Public Networking"**
4. Click **"Generate Domain"**
5. Copy your URL:

```
Example: https://ev-battery-backend-production.up.railway.app
```

---

### Step 7: Test Your API

Open in browser:
```
https://YOUR-BACKEND-URL.up.railway.app/docs
```

You'll see Swagger UI with all endpoints!

**Test with curl:**
```bash
# Health check
curl https://YOUR-URL.up.railway.app/health

# Send test telemetry
curl -X POST https://YOUR-URL.up.railway.app/api/telemetry \
  -H "Content-Type: application/json" \
  -d '{"vehicleId":"test-001","voltage":396.5,"current":45.2,"temperature":32.5,"internalResistance":42.8,"soc":78,"power":17.9,"isCharging":false,"cycleCount":100}'
```

---

### Step 8: Update Arduino/ESP32 Code

Edit `arduino/esp32_battery_monitor.ino`:

```cpp
// Line 12-13: Change these
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// Line 16: Change this
const char* serverUrl = "https://YOUR-BACKEND-URL.up.railway.app/api/telemetry";
```

Upload to ESP32!

---

### Step 9: Update React App

Edit `src/services/api.js`:

```javascript
// Line 10: Change this
const RAILWAY_API_URL = "https://YOUR-BACKEND-URL.up.railway.app";

// Line 13: Enable Railway
const USE_RAILWAY_API = true;
```

---

### Step 10: Run and Test!

```bash
# Terminal 1: Run React app
cd c:\Users\SriHaran\Documents\Projects\evapp
npm run dev

# Open browser
http://localhost:3000
```

Your app now fetches data from Railway + PostgreSQL!

---

## 🗄️ Database Schema

### Tables Created

| Table | Purpose |
|-------|---------|
| `vehicles` | Registered EVs |
| `telemetry` | Sensor readings from ESP32 |
| `predictions` | SOH, RUL, EUL predictions |
| `alerts` | System warnings |

### View Database

1. In Railway, click **PostgreSQL service**
2. Go to **"Data"** tab
3. Browse tables and data directly!

Or use **pgAdmin** / **DBeaver** with the connection string from Variables.

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/telemetry` | Receive ESP32 data |
| GET | `/api/telemetry/{vehicle_id}` | Get latest reading |
| GET | `/api/telemetry/{vehicle_id}/history` | Get history |
| GET | `/api/predict/soh/{vehicle_id}` | SOH prediction |
| GET | `/api/predict/rul/{vehicle_id}` | RUL prediction |
| GET | `/api/predict/eul/{vehicle_id}` | EUL prediction |
| GET | `/api/health/{vehicle_id}` | Health metrics |
| GET | `/api/alerts/{vehicle_id}` | Get alerts |
| GET | `/api/vehicles` | List all vehicles |

---

## 🧪 Test Locally (Optional)

```bash
cd c:\Users\SriHaran\Documents\Projects\evapp\backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run (uses SQLite locally)
python main.py
```

Open http://localhost:8000/docs

---

## 💰 Railway Pricing

| Plan | Cost | Limits |
|------|------|--------|
| **Hobby** | $5/month | 512MB RAM, 1GB disk |
| **Pro** | $20/month | Unlimited |

For your project, **Hobby plan is sufficient**!

---

## 🔧 Troubleshooting

### "Database connection failed"
- Check DATABASE_URL is set in Variables
- Ensure PostgreSQL service is running (green checkmark)

### "CORS error in browser"
- Backend allows all origins by default
- Check browser console for specific error

### "ESP32 not sending data"
- Verify WiFi credentials
- Check serverUrl matches Railway URL
- Test with Serial Monitor

---

## 📁 Files in Backend

```
backend/
├── main.py           # FastAPI app with all endpoints
├── models.py         # SQLAlchemy database models
├── database.py       # Database connection
├── requirements.txt  # Python dependencies
├── Procfile         # Railway start command
├── railway.json     # Railway config
└── README.md        # This file
```
