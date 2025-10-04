from dotenv import load_dotenv
import os

load_dotenv()

DOCKER_COMMAND = "docker"
SERVER_ARGS = ["mcp", "gateway", "run", "--filtering", "--filter-port", "8000", "--query-port", "8080"]
SERVER_ENV = {
    "LOCALAPPDATA": os.getenv("LOCALAPPDATA", "C:\\Users\\Default\\AppData\\Local"),
    "ProgramFiles": os.getenv("ProgramFiles", "C:\\Program Files"),
    "ProgramData": os.getenv("ProgramData", "C:\\ProgramData"),
}
MODEL_NAME = "llama-3.3-70b-versatile"
SERVER_QUERY_URL = os.getenv("SERVER_QUERY_URL", "http://localhost:8080/query")