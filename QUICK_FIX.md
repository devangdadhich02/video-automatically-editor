# 🔧 Quick Fix - API Key Error

## Problem
```
Error code: 401 - Incorrect API key provided
```

## Solution (3 Simple Steps)

### Step 1: Get New API Key
1. Open: https://platform.openai.com/api-keys
2. Click **"Create new secret key"**
3. Copy the key (it shows only once!)

### Step 2: Update .env File

**Option A: Using PowerShell (Easiest)**
```powershell
# Replace YOUR_NEW_KEY with the actual key you copied
"OPENAI_API_KEY=YOUR_NEW_KEY" | Out-File -FilePath .env -Encoding utf8 -NoNewline
```

**Option B: Using fix-api-key.ps1 script**
```powershell
.\fix-api-key.ps1
```

**Option C: Manual Edit**
1. Open `.env` file in Notepad
2. Replace the old key with new one:
   ```
   OPENAI_API_KEY=sk-proj-your-new-key-here
   ```
3. Save the file

### Step 3: Restart Docker
```powershell
docker-compose down
docker-compose up --build
```

## Verify It's Working

After restart, try uploading a video again. If it works, you'll see:
- ✓ Video processed! Transcript extracted...

## Still Getting Error?

Check these:
- ✅ API key starts with `sk-proj-` or `sk-`
- ✅ No extra spaces in the key
- ✅ Account has credits/balance
- ✅ API key is not expired/revoked

