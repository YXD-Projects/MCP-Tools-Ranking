# Youssefs-Playground

## Overview
This project is an MCP client structured in modules (`config`, `core`, `services`) that interacts with the Docker's MCP Gateway, to demonstrate the usage of the new Docker's MCP Gateway endpoint.

## Prerequisites
- Python 3.11 or higher
- Docker's MCP Gateway running and accessible

## Installation
1. Clone this repository or copy the files to your machine.
2. Create and activate a virtual environment:
   ```cmd
   (Windows)
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install Python dependencies:
   ```cmd
   export TRANSFORMERS_NO_TF=1
   pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
   pip install fastapi uvicorn pydantic faiss-cpu sentence-transformers numpy
   pip install -r requirements.txt
   ```

## Environment Configuration
The project uses environment variables, which can be set in a `.env` file at the project root. Example content:
```
// Paths related to run the docker's MCP server. 
LOCALAPPDATA=C:\Users\YourName\AppData\Local
ProgramFiles=C:\Program Files
ProgramData=C:\ProgramData
// The new query endpoint of the docker's MCP server. (Change the port if needed)
SERVER_QUERY_URL=http://localhost:8080/query
// API Key for running Groq models
GROQ_API_KEY=your_groq_api_key
```
If these variables are not set, default values are used (see `config/settings.py`).

## Running the Server or Services
- Docker and server parameters are configured in `config/settings.py`.
- To start the server, adapt the command as needed (see the `SERVER_ARGS` variable).
- To run the main project:
   ```cmd
   python main.py
   ```

## Important Environment Variables
- `LOCALAPPDATA`, `ProgramFiles`, `ProgramData`: Windows system paths
- `SERVER_QUERY_URL`: Server API URL

## Troubleshooting
- Make sure Docker and it's MCP Gateway is installed and accessible in your terminal.
- Ensure required ports (default 8080) are available.
- Check logs for any error messages.

## Project Structure
- `main.py`: main entry point
- `config/`: configuration and environment variables
- `core/`: main logic (client, server, query processing)
- `services/`: external services and tools
