from tkinter import *
from PIL import Image
from tkinter import ttk
import PIL.ImageOps   
import PIL.ImageGrab as ImageGrab
import numpy as np
import ctypes
import subprocess



# painting on painting canvas, brush radius can be adjusted
def paint(event):
    radius = 15
    x1,y1=(event.x-radius),(event.y-radius)
    x2,y2=(event.x+radius),(event.y+radius)
    canvasD.create_oval(x1,y1,x2,y2,fill="black")

# painting canvas reset
def reset_button_clicked():
    # data insertion
    for i in range(10):
        table.insert(parent="",index=i,values=(i,0))
    canvasD.delete("all")




# SAVE
# saving image from painting canvas as 8 bit grayscale 28x28 pixels, 
# invert the colors, flatten it into a vector, and save the vector to a text file with three decimal points
def ok_button_clicked():
    # Get device scaling factor
    scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

    # Grab the image from the canvas (280x280 pixels), adjusted for scaling
    img = ImageGrab.grab(bbox=(scaleFactor * canvasD.winfo_rootx(), scaleFactor * canvasD.winfo_rooty(), 
                              scaleFactor * (canvasD.winfo_rootx() + canvasD.winfo_width()), 
                              scaleFactor * (canvasD.winfo_rooty() + canvasD.winfo_height())))

    # Resize the image to 28x28 pixels and convert to grayscale
    img_resized = img.resize((28, 28)).convert("L")

    # Convert the image into a numpy array (uint8 type by default)
    img_array = np.asarray(img_resized)
    
    # Normalize the pixel values to be between 0 and 1 (float32)
    img_array_normalized = img_array.astype(np.float32) / 255.0
    
    # Invert the colors (MNIST has white digits on a black background, so we invert the image)
    img_array_inverted = 1.0 - img_array_normalized
    
    # Flatten the 28x28 array into a 1D array (vector)
    img_vector = img_array_inverted.flatten()
    
    # Print the flattened vector to the console (optional)
    #print(img_vector)
    
    # Save the vector to a text file as a single 784-long vector (1 line)
    with open("input_vector.txt", "w") as file:
       # Write each pixel value to a new line
       for val in img_vector:
           file.write(f"{val:.6f}\n")  # Write each pixel value with 6 decimal places
    
    # Optionally, save the processed image as a PNG file
    img_resized.save("image_temp.png")
    
    # Run the C++ executable
    subprocess.run(['callthis.exe'])
    
    
    # Open and read the output_vector.txt file
    with open("output_vector.txt", "r") as file:
        output_values = file.readlines()
    # Assuming the output vector has the same format, update the prediction column
    for i in range(10):
        prediction = float(output_values[i].strip())  # Convert each value to float
        table.insert(parent="",index=i,values=(i,prediction))
        

    
    


# main window parametres
app = Tk()
app.title("Handwritten digit recognition")
app.geometry("550x450")
app.resizable(0, 0)
app.tk.call("tk","scaling",2.0)

# main label
lb1 = Label(app, text="Draw a number from 0 to 9:")
lb1.grid(row = 0, column = 0, columnspan=2,sticky=N, pady=10, padx=10)

# painting canvas
canvasD = Canvas(app, bg="white", height="280", width="280")
canvasD.grid(row = 1, column = 0, columnspan=2, sticky=NW, pady=10, padx=10)

# ok button
ok_button = Button(app, text="Predict", fg="black", command=ok_button_clicked)
ok_button.grid(row = 2, column = 0, sticky=E, pady=10, padx=10)

# reset button
reset_button = Button(app, text="Clear", fg="black", command= reset_button_clicked)
reset_button.grid(row = 2, column = 1, sticky=W, pady=10, padx=10)

# mouse motion painting
canvasD.bind("<B1-Motion>", paint)

# table settings
table = ttk.Treeview(app, columns=("number","score"),show="headings")
table.heading("number", text="Digit:")
table.heading("score",text="Prediction:")
table.column("number", width=100, anchor=N)
table.column("score", width=100, anchor=N)












# data insertion
for i in range(10):
    table.insert(parent="",index=i,values=(i,0))


#table position
table.grid(row = 0, column = 2, rowspan=3, sticky=W, pady=10, padx=10)

app.mainloop()

