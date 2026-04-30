# AAI Coursework 2

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

`uv sync` reads the `pyproject.toml` file and installs all listed dependencies (including Streamlit).

### Running the App

```bash
streamlit run main.py
```

This will start the Streamlit development server and open the app in your browser.

### Adding new packages

```bash
uv add python-dotenv
```

### ENVs to add
OPENROUTER_API_KEY
