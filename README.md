# PDF Renamer using LLM

This Python script automates the renaming of PDF files by extracting key information (title, first author, and publication year) from the document's first page using a Large Language Model (LLM). It is designed to be used for research papers that are downloaded from sources such as arXiv.

## Features

* Extracts the title, first author, and year from the first page of PDF files using an LLM.
* Renames the PDF files using the extracted information.

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/ae-aydin/llm-renamer
    cd llm-renamer
    ```

2. **(Optional) Activate a Virtual Environment**

    If you are not in a virtual environment, it is recommended to use virtual environment for this project.

    ```bash
    python -m venv .venv
    ```

    ```bash
    source .venv/bin/activate # On Linux/macOS 
    .venv\Scripts\activate # On Windows
    ```

    or if you are using ***uv***:

    ```bash
    uv venv
    .venv\Scripts\activate
    ```

3. **Install dependencies**
    Run the following command to install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    or if you are using ***uv***:

    ```bash
    uv pip install -r requirements.txt
    ```

## Usage

1. Place PDF files: Put the PDF files that you want to rename into a directory.

2. Run the script: Open your terminal, and navigate to the folder containing *renamer.py* and *settings.toml*

3. Execute the script: You can run the script by using the following command in terminal:

```bash
python renamer.py --directory path/to/your/pdfs
```

Replace *path/to/your/pdfs* with the actual path to the directory containing your PDF files.

Note: When you first run, it might take a while to execute, as the script will be downloading the language model. Consequent runs will be faster.

## Limitations

* Only processes the first page of PDF files.

* Extraction relies on the LLM's accuracy, which is not always guaranteed.

* Extracts only the title, first author, and publication year.

## License

 This project is licensed under the Apache 2.0 License.
