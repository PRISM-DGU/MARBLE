# MARBLE

**M**ulti-**A**gent **R**easoning for **B**ioinformatics **L**earning and **E**volution

MARBLE is a LangGraph-based multi-agent system for automating drug response prediction research.

## Requirements

- Python 3.11+
- Docker
- Conda (recommended)

## Installation

### 1. Create and Activate Environment

```bash
conda create -n marble python=3.11 -y
conda activate marble
```

### 2. Install Dependencies

```bash
# Install runtime dependencies (required)
pip install -e ".[runtime]"

# Install development tools (optional)
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
python -c "import langgraph, langchain, docker; print('All packages installed successfully')"
```

## Configuration

### 1. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and configure:

```bash
# Required API Keys
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"

# Project Settings
PROJECT_ROOT="/path/to/MARBLE"
USER_ID=$(whoami)
```

## Docker Setup

### 1. Build MCP Server Images

```bash
./infrastructure/container_management_scripts/build-mcp-images.sh
```

### 2. Build Model Execution Images

```bash
./docker_images/build.sh
```

### 3. Start MCP Servers

```bash
./infrastructure/container_management_scripts/start-mcp.sh
```

## Running MARBLE

```bash
langgraph dev
```

## Project Structure

```
MARBLE/
├── agent_workflow/          # Core agent workflow logic
│   ├── main_graph_builder.py
│   ├── state.py
│   └── workflow_subgraphs/
├── configs/                 # Configuration files
├── docker_images/           # Drug response model containers
└── infrastructure/          # MCP server containers
```

## License

MIT License
