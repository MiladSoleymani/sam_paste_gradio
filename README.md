# sam_paste_gradio

This repository provides a tool for image segmentation using SAM (Segment Anything Model) and Gradio for the interface. The tool aims to facilitate the segmentation process with an intuitive UI.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
The `sam_paste_gradio` tool integrates SAM, a powerful image segmentation model, with Gradio, a user-friendly interface library, to create a seamless experience for segmenting images.

## Requirements
- Python 3.x
- Packages listed in `requirements.txt`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/MiladSoleymani/sam_paste_gradio.git
    cd sam_paste_gradio
    ```
2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage
To run the tool, use the following command:
```bash
python run.py
```
![2023-07-24 12 00 06](https://github.com/NaserFaryad/generative_defect_kyocera/assets/78655282/cf343329-c73d-43fd-b886-31fe0807193d)

## Repository Structure
- `sam`: Contains the implementation of the Segment Anything Model.
- `utils.py`: Utility functions for the project.
- `run.py`: Main script to run the application.
- `requirements.txt`: Lists the dependencies required to run the software.

## Contributing
We welcome contributions to improve `sam_paste_gradio`. Please fork the repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
