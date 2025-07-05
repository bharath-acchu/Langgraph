import gradio as gr
from graph import run_agent
from retrievr import load_and_index
import os
import shutil

def upload_file(file_path):
    

    # Extract just the filename
    filename = os.path.basename(file_path)

    # Destination in ./data
    dest_path = os.path.join("data", filename)

    # Copy file from Gradio temp to ./data
    shutil.copy(file_path, dest_path)

    # Index the file
    load_and_index(dest_path)

    return "✅ Document indexed and ready!"

def ask_question(query):
    return run_agent(query)

with gr.Blocks() as demo:
    gr.Markdown("### Agentic RAG (Gemini) with PDF/DOCX/TXT Support")

    with gr.Row():
        file_input = gr.File(label="Upload Document", type="filepath")
        upload_btn = gr.Button("Submit Document")
        upload_status = gr.Textbox(label="Status")

    upload_btn.click(fn=upload_file, inputs=[file_input], outputs=[upload_status])

    with gr.Row():
        query_input = gr.Textbox(placeholder="Ask a question about the document...")
        answer_output = gr.Textbox(label="Answer")
        query_btn = gr.Button("Ask")

    query_btn.click(fn=ask_question, inputs=[query_input], outputs=[answer_output])

demo.launch()
