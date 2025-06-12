# 📄 Line-by-Line OCR Text Extractor

A Streamlit application for extracting text from images line by line with bounding boxes, ideal for data extraction tasks.

## 🚧 Status

This project is currently under development. Features being worked on:

- Improving text extraction accuracy
- Adding support for more document formats
- Enhanced user interface
- Batch processing capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ installed
- Tesseract OCR installed (required for OCR functionality)
  - [Download Tesseract OCR](https://github.com/tesseract-ocr/tesseract#installing-tesseract) for your operating system
  - Make sure the Tesseract executable is in your system PATH

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/text-extractor-streamlit.git
cd text-extractor-streamlit
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit application with:

```bash
streamlit run streamlit_app.py
```

The application should open in your default web browser at http://localhost:8501.

## 📋 Features

- Extract text line by line from images with visual bounding boxes
- Edit extracted text directly in the interface
- Add and delete rows manually
- Preview cropped line images
- Export data to CSV or Excel formats
- Multiple image upload support

## 📁 Project Structure

```
.
├── README.md              # This file
├── streamlit_app.py       # Main application code
├── requirements.txt       # Python dependencies
├── demo/                  # Demo files and screenshots
└── samples/               # Sample images for testing
        ├── 1.png
        └── 2.png
```

## 📦 Dependencies

```
numpy==1.24.4
opencv-contrib-python==4.10.0.84
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
pandas==1.5.3
pytesseract==0.3.13
streamlit==1.45.1
Pillow
xlsxwriter
```

## 🙏 Acknowledgements

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for OCR capabilities
- [Streamlit](https://streamlit.io/) for the web interface
