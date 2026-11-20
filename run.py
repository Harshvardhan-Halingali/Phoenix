import uvicorn
import sys
import os

if __name__ == "__main__":
    print("==================================================================")
    print("           IMAGE TRACE — Forensic Analysis Platform               ")
    print("==================================================================")
    print("Starting server at http://127.0.0.1:8080 ...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True, log_level="info")
