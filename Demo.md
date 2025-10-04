# Demo Overview

This project demonstrates how an MCP client connects to Docker’s MCP Gateway and a FAISS-based filtering service to rank and return the most relevant tools for a user query.

#### The setup includes:
- mcp-client → Python client that sends queries and lists tools
- mcp-gateway → Docker plugin exposing the new MCP endpoint
- filter_controller → Ranking service using sentence-transformers + FAISS


#### Navigation
- [Client Setup](#client-setup)
- [Gateway Setup](#mcp-gateway-setup)
- [Running the Services](#running-the-services)
- [Demo Steps & Output](#demo-steps-and-output)


---
<br><br>


# Client Setup 

## Prerequisites:
- Python 3.11 +
- Docker Desktop running
- MCP Gateway installed (steps included)
<br><br>

# Steps (macOS/Linux/Windows):

### 1. Navigate to client
```cmd
cd MCP-Tools-Ranking/mcp-client
```

### 2. Create and Activate a Virtual Environment

#### Windows (PowerShell):
```cmd
python -m venv venv
.\venv\Scripts\activate
```

#### macOS / Linux:
```cmd
python3 -m venv .venv
source venv/bin/activate
```

### 3. Install dependencies
(macOS)
```cmd
pip install -r requirements_mac.txt 
```

```cmd
(Windows)
pip install -r requirements.txt 
```

- Deactivate venv after


### Environment variables (.env):
create a env file at project root
```cmd
touch .env
```

Paste the following configurations:
The new query endpoint of Docker's MCP server (adjust port if needed)
- SERVER_QUERY_URL=http://localhost:8080/query

API Key for running Groq models (if used)
- GROQ_API_KEY=your_groq_api_key


### Run the client:
#### Windows:
```cmd
python -m mcp_client.main
```

#### MacOS:
```cmd
python3 -m mcp_client.main
```

Expected: it loads configs, connects to the Docker MCP Gateway, and lists available MCP servers.
Susupend service once confirmed it works



---
<br><br>


# MCP Gateway Setup

The MCP Gateway acts as the bridge between client and MCP servers.

### 1. — Make Sure Docker Is Installed and Running

#### Install Docker Desktop (if you haven’t already).

#### Windows: 
- https://docs.docker.com/desktop/setup/install/windows-install/
  
#### Mac: 
- https://docs.docker.com/desktop/setup/install/mac-install/ 

Once installed:
- Open Docker Desktop and verify it’s running.
- Navigate to settings and find "Beta Features". In that screen you will enable MCP Docker Toolkit
- In the "MCP Toolkit" you can add servers with tools to be tested
    - Some tools may require API keys, personal access token or OAuth (Github for example)
- Naviagate to mcp-gateway directory

### 2. — Remove Any Old MCP Plugin Versions

#### To prevent version conflicts, delete any existing Docker CLI plugin folder:

#### Windows:
```cmd
Remove-Item -Recurse -Force "$env:USERPROFILE\.docker\cli-plugins" -ErrorAction SilentlyContinue
```

#### Mac:
```cmd
rm -rf ~/.docker/cli-plugins
```

### 3. — Create a cli-plugins Folder/Docker executable/Move excutable

#### Now recreate the folder:

#### Windows PowerShell
```cmd
mkdir "$env:USERPROFILE\.docker\cli-plugins"
```

#### macOS / Linux
```cmd
mkdir -p "$HOME/.docker/cli-plugins/"
```

#### Build the docker-mcp Executable:
#### From the root of the mcp-gateway project directory, run:
```cmd
go build -o ./dist/docker-mcp.exe ./cmd/docker-mcp
```

This command:
- Uses Go to compile the MCP Gateway source code
- Outputs the executable file to the dist folder
- Names it docker-mcp.exe (so Docker recognizes it as a plugin)


#### Move the new file to your Docker CLI plugin directory:
#### Windows PowerShell
```cmd
copy "dist\docker-mcp.exe" "$env:USERPROFILE\.docker\cli-plugins\"
```

#### macOS / Linux
```cmd
cp dist/docker-mcp "$HOME/.docker/cli-plugins/"
```

### Step 4 — Verify Installation

#### Run this command to confirm Docker recognizes the plugin:
```cmd
docker mcp --help
docker mcp gateway run --help
```
- If everything was set up correctly, you’ll see usage instructions for the docker mcp command.



---
<br><br>


## Running the Services

Once Docker Desktop is installed and actively running in the background.
You’ll be using your system terminal (PowerShell on Windows or Terminal on macOS/Linux).


### 1. — Navigate to the Project Folder

Enter the client folder where the filtering service is located and start your vitual enviroment:
```cmd
cd mcp-client
```

### 2. — Run the Filtering Service

Once you’re inside the mcp-client directory and your virtual environment is active, start the filtering service:
```cmd
python filtering_service.py
```

- Keep this terminal window open the service needs to keep running in the background.
- The service may need up to 30s to startup

#### Find and kill previous process:
```cmd
lsof -i :8000
kill -9 <PID>
```

### 3. — Run the MCP Client

Open a second terminal window from the root directory and run:
#### Windows:
```cmd
python -m mcp_client.main
```
#### MacOS:
```cmd
python3 -m mcp_client.main
```



---


# Demo Steps and Output

1. Make sure Docker Desktop is running.
   
2. Run the Filtering Service:
```cmd
cd mcp-client
python filtering_service.py
```
Kill previous process if needed:
```cmd
lsof -i :8000
kill -9 <PID>
```

#### In the terminal you will see:

When the filtering_service.py is started, you should see logs similar to the output below in your terminal.
This confirms that the SentenceTransformer model was loaded correctly and the Uvicorn server is running locally.

<img width="688" height="148" alt="Screenshot 2025-10-04 at 3 34 28 PM" src="https://github.com/user-attachments/assets/4ddf407c-1ee1-4988-94be-bc085ca8d176" />

#### What this means:
- Model loaded successfully: The all-MiniLM-L6-v2 embedding model is initialized and ready to encode inputs.
- Server started: Uvicorn is serving your API at http://localhost:8000
- To stop: Press CTRL + C in the terminal window.




3. Run the MCP Client in the root directory (MCP-Tools-Ranking):
#### Windows:
```cmd
python -m mcp_client.main
```

#### MacOS:
```cmd
python3 -m mcp_client.main
```

#### In the terminal you will see:

If everything is configured correctly, you’ll see logs similar to the following:
- You’ll then see that multiple MCP servers are enabled and the client begins to list and run each container:

<p align="center">
  <img width="1215" height="532" alt="Screenshot 2025-10-04 at 4 46 28 PM" src="https://github.com/user-attachments/assets/7afafd95-aff8-4a27-879c-86b35daacfb2" />
</p>
<p align="center"><em>Figure 1 – MCP Client connecting to all Docker MCP servers and listing tools.</em></p>

<p align="center">
  <img width="608" height="119" alt="Screenshot 2025-10-04 at 4 46 38 PM" src="https://github.com/user-attachments/assets/e690c88d-1b8e-487d-b370-dee4b97dea94" />
</p>
<p align="center"><em>Figure 2 – Tool indexing and filtering completed successfully.</em></p>



#### After a Query:

The MCP Client should successfully execute the create_repository tool from the GitHub MCP server.

#### The client logs indicate the full tool call sequence:
- It scans the tool call arguments ({'name': 'my_repo'}) for any secrets.
- It securely runs the GitHub MCP container (ghcr.io/github/github-mcp-server) using the local Docker environment.
- The server processes the request and returns a successful response.

<p align="center">
  <img width="579" height="60" alt="Screenshot 2025-10-04 at 4 47 02 PM" src="https://github.com/user-attachments/assets/56b2be3d-2433-4d92-80af-f4778fcf02ed" />
  <img width="1222" height="252" alt="Screenshot 2025-10-04 at 4 47 23 PM" src="https://github.com/user-attachments/assets/ce47eeee-bb77-4948-a901-0bf761b77369" />
</p>
<p align="center"><em>Figure 4 – Successful tool call: a new GitHub repository (<code>my_new_newest_repo</code>) is created.</em></p>


#### In the server terminal you will see logs:
- Server active: Uvicorn service is running locally.
- Model indexed: 117 total tools embedded and clustered.
- API calls successful: Indexing and filtering endpoints are responding with 200 OK.
- Pipeline operational: Confirms live end-to-end communication between the MCP Client and the filtering service.

<p align="center">
  <img width="1360" height="387" alt="Screenshot 2025-10-04 at 4 55 06 PM" src="https://github.com/user-attachments/assets/4ad64b5f-731c-4518-b62d-d62332d8f3cc" />
</p>
<p align="center"><em>Figure 5 – Filtering service running successfully: tools embedded (n=117), indexed with IVF-Flat(IP), and confirmed via 200 OK API responses.</em></p>


#### This means:
- The GitHub MCP server is responding properly through the Docker MCP Gateway.
- GitHub personal access token is authenticating successfully.
- A new repository named my_repo was created directly from the MCP Client interface.
- The MCP Client remains active and ready for follow-up actions (e.g., adding files, branches, or issues).




