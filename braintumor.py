from tkinter import *
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from joblib import load
from PIL import Image, ImageTk

model = load('ML/polynomial linear regression/Brain_tumor_model.joblib')

root = Tk()
root.title("Brain Tumor Detection")
root.geometry("1150x620")
root.config(bg="#d8e7d4")
root.resizable(False, False)

img_path = ""  

def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Error", "Failed to read the image. Please select a valid image.")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((400, 400))  
        img_tk = ImageTk.PhotoImage(img_pil)

        image_label.config(image=img_tk)
        image_label.image = img_tk  
        image_label.place(x=100, y=175)
    else:
        messagebox.showwarning("No File", "No image file selected.")

def predict():
    if not img_path:
        messagebox.showerror("Error", "Please upload an image first!")
        return
    
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img_resized = cv2.resize(img_gray, (200, 200))  
    img_flattened = img_resized.flatten()  
    img_scaled = img_flattened / 255.0 

    prediction = model.predict(img_scaled.reshape(1, -1)) 
    result = "Tumor" if prediction[0] == 1 else "No Tumor"
 
    result_label.config(text=f"Prediction: {result}")
     
    img_pred_resized = cv2.resize(img_resized, (400, 300))  
    img_pred_pil = Image.fromarray(img_pred_resized)
    img_pred_tk = ImageTk.PhotoImage(img_pred_pil)

    pred_image_label.config(image=img_pred_tk)
    pred_image_label.image = img_pred_tk  
    pred_image_label.place(x=650, y=175)

heading = Label(root, text="Brain Tumor Detection", font="Corbel 25 bold", fg="black", bg="#d8e7d4")
heading.pack(pady=20)

frame1 = Frame(root, width=500, height=500, bg="#f2f2e9")
frame1.place(x=50, y=90)

upload_button = Button(frame1, text="Upload Image", command=upload_image, cursor="hand2",width=20, font="arial 18 bold", fg="black", bg="#d8e7d4", activebackground="#e8fde3", activeforeground='black')
upload_button.place(x=90, y=20)

frame2 = Frame(root, width=500, height=500, bg="#f2f2e9")
frame2.place(x=600, y=90)

predict_button = Button(frame2, text="Tumor Detection", command=predict, cursor="hand2",width=20, font="arial 18 bold", fg="black", bg="#d8e7d4", activebackground="#e8fde3", activeforeground='black')
predict_button.place(x=90, y=20)

image_label = Label(root)
image_label.place(x=50, y=100)

result_label = Label(root, text="", font=("Arial", 16),bg="#f2f2e9",fg="black")
result_label.place(x=680, y=520) 

pred_image_label = Label(root)
pred_image_label.place(x=50, y=100)

root.mainloop()
