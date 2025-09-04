import streamlit as st
from docx import Document  # For reading .docx files
import mammoth             # For reading .doc files
import textract            # Fallback for more complex formats

def read_docx(file):
    """Reads text from a .docx file."""
    document = Document(file)
    text = "\n".join([para.text for para in document.paragraphs])
    return text

def read_doc(file):
    """Reads text from a .doc file using mammoth."""
    with file as f:
        result = mammoth.extract_raw_text(f)
    return result.value

def read_fallback(file):
    """Fallback for reading text using textract."""
    return textract.process(file).decode("utf-8")

def main():
    st.title("Resume Text Extractor")
    uploaded_file = st.file_uploader("Upload a resume file (.doc or .docx)", type=["doc", "docx"])

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        text = ""
        
        try:
            if file_type == "docx":
                text = read_docx(uploaded_file)
            elif file_type == "doc":
                text = read_doc(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.warning("Attempting fallback method...")
            try:
                text = read_fallback(uploaded_file)
            except Exception as e:
                st.error(f"Fallback method failed: {e}")
        
        if text:
            st.subheader("Extracted Text")
            st.text_area("Resume Content", text, height=400)
        else:
            st.error("Failed to extract text from the file.")

if __name__ == "__main__":
    main()
