import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

print("Loading model...")
model = tf.keras.models.load_model('best_model_xception.keras')
print("Model loaded successfully.")

# Get model input shape dynamically
input_shape = model.input_shape[1:3]
if input_shape[0] is None:
    input_shape = (299, 299)

def predict_image(img):
    img = img.resize(input_shape)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    
    # Check if binary (sigmoid) or multi-class (softmax)
    if predictions.shape[-1] == 1:
        prob = float(predictions[0][0])
        # Assuming 0 is Cat and 1 is Dog (alphabetical order)
        return {"고양이 (Cat)": 1.0 - prob, "강아지 (Dog)": prob}
    else:
        prob_cat = float(predictions[0][0])
        prob_dog = float(predictions[0][1])
        return {"고양이 (Cat)": prob_cat, "강아지 (Dog)": prob_dog}

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Cat vs Dog Classifier 🐱🐶",
    description="Upload an image of a cat or dog, and the AI model will tell you which one it is!"
)

if __name__ == "__main__":
    interface.launch(share=True)
