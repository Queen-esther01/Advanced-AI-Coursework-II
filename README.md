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

**Indexing** (`doc_parser`, `slide_renderer`):

```bash
uv add python-docx python-pptx
```

**macOS**

```bash
brew install --cask libreoffice
brew install poppler libemf2svg librsvg
```

**Linux (Debian/Ubuntu)**

```bash
sudo apt install libreoffice poppler-utils libemf2svg librsvg2-bin
```

**Windows** — [LibreOffice](https://www.libreoffice.org/download/) + Poppler on PATH (`pdftoppm`); libemf2svg/librsvg if indexing WMF/EMF images.

- **LibreOffice** + **poppler** — CPT slide renders (`soffice`, `pdftoppm`)
- **libemf2svg** + **librsvg** — WMF/EMF → PNG in docx/pptx (or Ghostscript / ImageMagick)

### Running the App

```bash
python -m streamlit run main.py
or
streamlit run main.py
```

This will start the Streamlit development server and open the app in your browser.

### Adding new packages

```bash
uv add python-dotenv
```

### ENVs to add
OPENROUTER_API_KEY

### Download the compressed .store directory and extract it into the root directory
