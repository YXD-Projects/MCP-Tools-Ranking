# MCP-Tools-Ranking

# Overview
This project consists of an MCP client connected to Docker’s MCP Gateway, and a ranking service that uses Sentence-Transformers and FAISS to rank and filter tool descriptions based on their
relevance to an incoming query. It addresses the problem of context-window overload, where too many tool schemas fill the LLM’s context window. By pre-filtering and returning only the most relevant tools, it reduces token usage, improves accuracy, and speeds up tool selection.


# Prerequisites
- Python 3.11 or higher
- Docker's MCP Gateway running and accessible

# Installation
1. Clone this repository.
2. For the client Setup check the [mcp-client README](mcp-client/README.md).
3. For the mcp-gateway:
   * Make Sure Docker Desktop is installed and running.
   * To make sure there are no conflicts, go to C:/Users/YourName/.docker and delete the cli-plugins folder if it exists.
   * Run these commands in your terminal:
   ```cmd
    # Create a new cli-plugins directory
    mkdir -p "$HOME/.docker/cli-plugins/"
    # Build and create a docker-mcp.exe
    go build -o ./dist/docker-mcp.exe ./cmd/docker-mcp
    # copy the docker-mcp.exe into the cli-plugins
    copy "dist\docker-mcp.exe" "$env:USERPROFILE\.docker\cli-plugins\"
   ```

# Running the Services
1. Make sire Docker Desktop is running.
2. Run the Filtering Service:
   ```cmd
   cd mcp-client
   python filtering_service.py
   ```
3. Run the MCP Client:
   ```cmd
   python main.py
   ```



