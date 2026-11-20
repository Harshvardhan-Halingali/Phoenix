import os
import sys
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() in ("true", "1")
    print("==================================================================")
    print("           IMAGE TRACE — Forensic Analysis Platform               ")
    print(f"Starting server at http://{host}:{port} ...")
    print("==================================================================")
    uvicorn.run("app.main:app", host=host, port=port, reload=debug, log_level="info")
