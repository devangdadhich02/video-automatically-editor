# OpenAI API Key Setup Instructions

## Issue
You're getting this error:
```
Error code: 401 - Incorrect API key provided
```

This means your API key is either:
- Invalid
- Expired
- Revoked
- Doesn't have proper permissions

## Solution

### Step 1: Get a New API Key

1. Go to: https://platform.openai.com/api-keys
2. Login to your OpenAI account
3. Click **"Create new secret key"**
4. Give it a name (e.g., "Video Editor")
5. Copy the key immediately (you won't see it again!)

### Step 2: Update .env File

**Option A: Using PowerShell Script**
```powershell
.\update-api-key.ps1
```

**Option B: Manual Update**
1. Open `.env` file in a text editor
2. Replace the old API key with your new one:
```
OPENAI_API_KEY=your-new-api-key-here
```
3. Save the file

**Option C: Using PowerShell Command**
```powershell
"OPENAI_API_KEY=your-new-api-key-here" | Out-File -FilePath .env -Encoding utf8 -NoNewline
```

### Step 3: Restart Docker Container

```powershell
docker-compose down
docker-compose up --build
```

## Verify API Key is Working

After restarting, try uploading a video again. If you still get errors, check:

1. **API Key Format**: Should start with `sk-proj-` or `sk-`
2. **API Key Length**: Should be around 50+ characters
3. **Account Status**: Make sure your OpenAI account has credits
4. **API Access**: Ensure you have access to Whisper API

## Check API Key in Docker

To verify the key is loaded in Docker:
```powershell
docker-compose config | Select-String -Pattern "OPENAI_API_KEY"
```

## Need Help?

- OpenAI API Documentation: https://platform.openai.com/docs
- Check API Status: https://status.openai.com/
- Verify Account Credits: https://platform.openai.com/account/usage

