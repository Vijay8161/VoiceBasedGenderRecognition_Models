import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT
    uvicorn.run("app.api:app", host="0.0.0.0", port=port)
