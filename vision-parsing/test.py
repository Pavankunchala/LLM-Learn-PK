from vision_parse import VisionParser, PDFPageConfig

#pip install vision-parse 

page_config = PDFPageConfig(
    dpi=400,
    color_space="RGB",
    include_annotations=True,
    preserve_transparency=False
)

# Initialize parser
parser = VisionParser(
    model_name="llama3.2-vision:11b", # For local models, you don't need to provide the api key
    temperature=0.4,
    top_p=0.3,
    extraction_complexity=False, # Set to True for more detailed extraction
    page_config=page_config
)

# Convert PDF to markdown
pdf_path = "jetson.pdf"
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")