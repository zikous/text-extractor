import streamlit as st
import pandas as pd
import os
import io
from PIL import Image
import pytesseract
import uuid
import cv2
import numpy as np


def line_extraction_with_bbox(image_path):
    """Extract text line by line with bounding boxes"""
    try:
        # Open image
        image = Image.open(image_path)
        img_array = np.array(image)
        original_img = img_array.copy()  # Keep original for cropping

        # Convert to RGB if needed (for OpenCV)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            original_img = img_array.copy()
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            original_img = img_array.copy()

        # Enhanced Tesseract configuration for line detection
        custom_config = "--oem 3 --psm 3"

        # Get detailed OCR data including bounding boxes
        ocr_data = pytesseract.image_to_data(
            img_array, config=custom_config, output_type=pytesseract.Output.DICT
        )

        extracted_lines = []
        current_line = []
        line_bbox = None  # Will store [x_min, y_min, x_max, y_max]

        # Process each detected text element
        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            if (
                text and int(ocr_data["conf"][i]) > 60
            ):  # Only consider confident detections
                # Get bounding box coordinates
                x, y, w, h = (
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["width"][i],
                    ocr_data["height"][i],
                )

                # Initialize or expand line bounding box
                if line_bbox is None:
                    line_bbox = [x, y, x + w, y + h]
                else:
                    line_bbox[0] = min(line_bbox[0], x)  # x_min
                    line_bbox[1] = min(line_bbox[1], y)  # y_min
                    line_bbox[2] = max(line_bbox[2], x + w)  # x_max
                    line_bbox[3] = max(line_bbox[3], y + h)  # y_max

                current_line.append(text)

                # Check if this is the last word in the line
                if (
                    i == len(ocr_data["text"]) - 1
                    or ocr_data["line_num"][i] != ocr_data["line_num"][i + 1]
                ):
                    # Combine words into a line
                    line_text = " ".join(current_line)
                    bbox_str = f"{line_bbox[0]},{line_bbox[1]},{line_bbox[2]-line_bbox[0]},{line_bbox[3]-line_bbox[1]}"

                    # Create cropped image of just this line
                    cropped_img = original_img[
                        max(0, line_bbox[1] - 5) : min(
                            original_img.shape[0], line_bbox[3] + 5
                        ),
                        max(0, line_bbox[0] - 5) : min(
                            original_img.shape[1], line_bbox[2] + 5
                        ),
                    ]

                    # Draw rectangle on full image for visualization
                    cv2.rectangle(
                        img_array,
                        (line_bbox[0], line_bbox[1]),
                        (line_bbox[2], line_bbox[3]),
                        (0, 255, 0),
                        2,
                    )

                    extracted_lines.append(
                        {
                            "text": line_text,
                            "bbox": bbox_str,
                            "full_image_with_bbox": img_array.copy(),
                            "cropped_image": cropped_img,
                        }
                    )

                    # Reset for next line
                    current_line = []
                    line_bbox = None

        return extracted_lines

    except Exception as e:
        raise Exception(f"OCR processing failed: {str(e)}")


def main():
    """Main application entry point for Streamlit UI"""

    # Initialize session state variables
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(
            columns=[
                "id",
                "filename",
                "text",
                "bbox",
                "full_image_with_bbox",
                "cropped_image",
            ]
        )
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = True
    if "edited_rows" not in st.session_state:
        st.session_state.edited_rows = {}
    if "rows_to_delete" not in st.session_state:
        st.session_state.rows_to_delete = set()
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = {}
    if "export_filename" not in st.session_state:
        st.session_state.export_filename = "extracted_text"

    # App title and description
    st.title("📄 Line-by-Line OCR Text Extractor")
    st.markdown("Upload images to extract text line by line with bounding boxes.")

    # Action buttons at the top
    if not st.session_state.df.empty:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            # Add new empty row
            if st.button("➕ Add Row", use_container_width=True):
                new_id = str(uuid.uuid4())
                new_row = pd.DataFrame(
                    {
                        "id": [new_id],
                        "filename": ["manual_entry"],
                        "text": [""],
                        "bbox": ["0,0,0,0"],
                        "full_image_with_bbox": [None],
                        "cropped_image": [None],
                    }
                )
                st.session_state.df = pd.concat(
                    [new_row, st.session_state.df], ignore_index=True
                )
                st.session_state.edited_rows[new_id] = ""
                st.rerun()

        with col2:
            # Show uploader again
            if st.button("📤 Upload More", use_container_width=True):
                st.session_state.show_uploader = True
                st.rerun()

        st.markdown("---")

    # Show uploader only if flag is True and no data exists
    if st.session_state.show_uploader or st.session_state.df.empty:
        with st.expander("📤 Upload Images", expanded=True):
            uploaded_files = st.file_uploader(
                "Drop image files here (PNG, JPG, etc.)",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                accept_multiple_files=True,
                key="file_uploader",
                label_visibility="collapsed",
            )

            if uploaded_files:
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    for uploaded_file in uploaded_files:
                        try:
                            # Store the uploaded image for preview
                            st.session_state.uploaded_images[uploaded_file.name] = (
                                uploaded_file.getvalue()
                            )

                            # Save the uploaded file temporarily
                            temp_path = f"temp_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Process the image
                            extracted_data = line_extraction_with_bbox(temp_path)

                            if extracted_data:
                                # Create new rows for this file
                                new_rows = []
                                for item in extracted_data:
                                    new_row = {
                                        "id": str(uuid.uuid4()),
                                        "filename": uploaded_file.name,
                                        "text": item["text"],
                                        "bbox": item["bbox"],
                                        "full_image_with_bbox": item[
                                            "full_image_with_bbox"
                                        ],
                                        "cropped_image": item["cropped_image"],
                                    }
                                    new_rows.append(new_row)
                                    st.session_state.edited_rows[new_row["id"]] = (
                                        new_row["text"]
                                    )

                                # Add to DataFrame at the top (newer entries first)
                                new_df = pd.DataFrame(new_rows)
                                st.session_state.df = pd.concat(
                                    [new_df, st.session_state.df], ignore_index=True
                                )

                            # Remove temp file
                            os.remove(temp_path)

                        except Exception as e:
                            st.error(
                                f"Failed to process image {uploaded_file.name}: {str(e)}"
                            )

                # Hide uploader after processing
                st.session_state.show_uploader = False
                st.rerun()

    # Show imported images preview section
    if st.session_state.uploaded_images:
        with st.expander("📷 Imported Images Preview", expanded=False):
            cols = st.columns(3)  # 3 images per row
            col_index = 0

            for filename, image_bytes in st.session_state.uploaded_images.items():
                with cols[col_index]:
                    st.image(image_bytes, caption=filename, use_container_width=True)
                    col_index = (col_index + 1) % 3

    # Main table interface
    st.markdown("### ✏️ Extracted Lines")

    if not st.session_state.df.empty:
        # Custom table header
        with st.container():
            cols = st.columns([1, 4, 1, 1])
            with cols[0]:
                st.markdown("**Source**")
            with cols[1]:
                st.markdown("**Text Content**")
            with cols[2]:
                st.markdown("**Preview**")
            with cols[3]:
                st.markdown("**Actions**")
        st.divider()

        # Custom table rows
        for index, row in st.session_state.df.iterrows():
            row_id = row["id"]

            # Skip if marked for deletion
            if row_id in st.session_state.rows_to_delete:
                continue

            # Initialize edited text if not exists
            if row_id not in st.session_state.edited_rows:
                st.session_state.edited_rows[row_id] = row["text"]

            with st.container():
                cols = st.columns([1, 4, 1, 1])

                # Source column
                with cols[0]:
                    st.text(row["filename"])

                # Text content column (editable)
                with cols[1]:
                    new_text = st.text_input(
                        "Text",
                        value=st.session_state.edited_rows[row_id],
                        key=f"text_{row_id}",
                        label_visibility="collapsed",
                    )
                    if new_text != st.session_state.edited_rows[row_id]:
                        st.session_state.edited_rows[row_id] = new_text
                        st.session_state.df.at[index, "text"] = new_text

                # Preview column
                with cols[2]:
                    if row["cropped_image"] is not None:
                        # Show the cropped image preview
                        cropped_img = Image.fromarray(row["cropped_image"])
                        st.image(
                            cropped_img,
                            caption="Extracted line",
                            use_container_width=True,
                        )
                    else:
                        st.text("No image")

                # Actions column
                with cols[3]:
                    if st.button("🗑️", key=f"delete_{row_id}"):
                        st.session_state.rows_to_delete.add(row_id)
                        st.rerun()

            st.divider()

        # Handle row deletions after rendering all rows
        if st.session_state.rows_to_delete:
            st.session_state.df = st.session_state.df[
                ~st.session_state.df["id"].isin(st.session_state.rows_to_delete)
            ].reset_index(drop=True)
            st.session_state.rows_to_delete = set()
            st.rerun()

        # Export options
        st.markdown("---")
        st.markdown("### 📤 Export Options")

        # Filename input
        st.session_state.export_filename = st.text_input(
            "Export filename (without extension)",
            value=st.session_state.export_filename,
            help="Enter the filename you want to use for the exported file",
        )

        # Create two columns for export buttons
        col1, col2 = st.columns(2)

        with col1:
            # Export to CSV
            csv = st.session_state.df[["text"]].drop_duplicates().to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                f"{st.session_state.export_filename}.csv",
                "text/csv",
                key="download-csv",
                use_container_width=True,
            )

        with col2:
            try:
                # Export to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    st.session_state.df[["text"]].drop_duplicates().to_excel(
                        writer, index=False, sheet_name="Extracted Text"
                    )

                excel_data = output.getvalue()
                st.download_button(
                    "Download as Excel",
                    excel_data,
                    f"{st.session_state.export_filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download-excel",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Excel export error: {str(e)}")
                st.info("Try installing: pip install xlsxwriter")

    else:
        st.info("No text extracted yet. Upload images to get started.")


if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Line-by-Line OCR Text Extractor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Hide Streamlit menu and footer
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    main()
