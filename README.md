# Pattern-Recognition-Project

### **Simple Workflow Explanation**

#### **1. Getting the Picture Data Ready**
- Used **Roboflow** (a free online tool) to:
  - Upload approximately 50 car pictures for each brand
  - Draw bounding boxes around logos with label (e.g. Honda_Logo, Mazda_Logo)
  - Split into 3 folders:
    - **Train** 
    - **Valid** 
    - **Test** 

#### **2. Teaching the Computer**
- Ran the **train_car_logo_detector.py** program to:
  - Show the computer all the labeled pictures
  - Let it practice 50 times (epochs)
  - Save what it learned as **best.pt** (its "brain" file)

#### **3. Finding and Cutting Out Logos**
- Used the **crop_car_logos.py** program to:
  - Load the "brain" file (**best.pt**)
  - Look at new car pictures one by one
  - Find logos (even if they're small or blurry)
  - Cut them out and save as new pictures

#### **4. Final Result**
- Got a folder full of:
  - Cropped logos
  - Named like: **"car123_logo_0.jpg"**
  - Ready for the next step in your project

