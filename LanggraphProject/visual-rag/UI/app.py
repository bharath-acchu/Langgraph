# ui/app.py
import gradio as gr
import os
from react_agent_graph import build_react_graph

graph = build_react_graph()

def process_image(image, question):

    if image is None or not question.strip():
        return "Please upload an image and enter a question."
    

    import uuid
    import os

    # Save uploaded image
    os.makedirs("image_store/images", exist_ok=True)

    # Create a unique filename
   # filename = f"{uuid.uuid4().hex}.jpg"
    #save_path = os.path.join("image_store/images", filename)
    # Save image to fixed path only once
    save_path = "image_store/images/temp.jpg"
    if not os.path.exists(save_path):
        image.save(save_path)
    
    #image.save(save_path)
    if image is None or not question.strip():
        return "Please upload an image and enter a question."

    inputs = {
        "image_path": save_path,
        "question": question,
        "status": "CONTINUE"
    }

    result = graph.invoke(inputs)
    return result["final_answer"]

def run_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Visual RAG Agent with Gemini (Gradio)")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Diagram")
            question_input = gr.Textbox(label="Ask a question")
        with gr.Row():
            output = gr.Textbox(label="Gemini's Answer")
        if image_input and question_input:
            submit_btn = gr.Button("Ask Gemini")

            submit_btn.click(fn=process_image, 
                            inputs=[image_input, question_input],
                            outputs=output)
    return demo
