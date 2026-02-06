import gradio as gr
import tensorflow as tf
import numpy as np

# Load model (Handling both potential file names just in case)
try:
    model = tf.keras.models.load_model('three_digit_multicolor_generator.keras')
except:
    model = tf.keras.models.load_model('generator_model.h5')

def generate_image(d1, d2, d3, c1, c2, c3):
    LATENT_DIM = 100 
    
    color_map = {"Red": 0, "Green": 1, "Blue": 2}
    
    c1_idx = color_map.get(c1, 0)
    c2_idx = color_map.get(c2, 0)
    c3_idx = color_map.get(c3, 0)

    # Prepare inputs
    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    
    label_d1 = np.array([[int(d1)]])
    label_d2 = np.array([[int(d2)]])
    label_d3 = np.array([[int(d3)]])
    
    label_c1 = np.array([[int(c1_idx)]])
    label_c2 = np.array([[int(c2_idx)]])
    label_c3 = np.array([[int(c3_idx)]])

    # Predict
    generated_image = model.predict([noise, label_d1, label_d2, label_d3, label_c1, label_c2, label_c3])

    # Scale to [0, 1]
    img = (generated_image[0] * 0.5) + 0.5
    
    return img

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 3-Digit Multicolor Generator")
    gr.Markdown("Generate a sequence of 3 digits, where **each digit** has its own specific color.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Digit 1")
            d1_val = gr.Slider(0, 9, step=1, label="Digit", value=1)
            c1_val = gr.Radio(["Red", "Green", "Blue"], label="Color", value="Red")
        
        with gr.Column():
            gr.Markdown("### Digit 2")
            d2_val = gr.Slider(0, 9, step=1, label="Digit", value=2)
            c2_val = gr.Radio(["Red", "Green", "Blue"], label="Color", value="Green")
            
        with gr.Column():
            gr.Markdown("### Digit 3")
            d3_val = gr.Slider(0, 9, step=1, label="Digit", value=3)
            c3_val = gr.Radio(["Red", "Green", "Blue"], label="Color", value="Blue")

    btn = gr.Button("Generate Image", variant="primary")
    
    # Fixed: Removed shape argument
    output = gr.Image(label="Generated Result") 

    btn.click(fn=generate_image, 
              inputs=[d1_val, d2_val, d3_val, c1_val, c2_val, c3_val], 
              outputs=output)

demo.launch()