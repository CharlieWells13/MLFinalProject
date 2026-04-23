# Final Project Deliverable I Outline

## 1. Group Information

### 1.1 Member 1
- **Name:** Charlie Wells
- **Major:** Computer Science

### 1.2 Member 2
- **Name:** Gihwan (Finn) Jung
- **Major:** Computer Science

### 1.3 Member 3
- **Name:** Aleksandre Khvadagadze
- **Major:** Computer Science

### 1.4 Group ID
- **Group ID:** Aleksandre-Charlie-Finn-Option2

## 2. Group Communication

### 2.1 Preferred Communication Method
- **Preferred method(s):** Text

### 2.2 Summary of Group Discussion
We will communicate via text regarding the project. We plan on meeting weekly to ensure the project is on course. We plan to check messages as soon as possible.

## 3. Collaboration Experience and Expectations

### 3.1 Prior Collaboration Experience
For collaborative work, we all agree that communication is the biggest part of making the project go well. Examples of bad communication include things like not responding to texts or waiting until the last minute to do work without telling anyone, which we all have experienced from other group members.

### 3.2 Concerns About Collaboration
We have different class schedules, so we may have scheduling conflicts. In this case, we can conduct our meetings virtually via Zoom. We also acknowledge that we may have different background knowledge on machine learning, and we will coordinate this by helping each other when necessary.

### 3.3 Preferred Approach for Collaboration
We plan on dividing work clearly by task and setting deadlines for each part. While doing so, we will share progress regularly and review each other’s work before submission.

## 4. Project Selection

### 4.1 Selected Project
Our group plans to work on **Project Option 2: Develop End-to-End ML Pipeline for Object Localization**.

## 5. Potential ML End-to-End Workflow

### 5.1 Problem Overview
The goal of this project is to build a machine learning pipeline that can identify and localize objects in images via the bounding boxes included in the data.

### 5.2 Proposed Workflow
1. Download the dataset  
   Obtain the object localization dataset and review its structure.
2. Organize and inspect the data  
   Check image formats, annotation files, and class labels.
3. Preprocess the data  
   Resize images, normalize them if needed, and clean invalid or inconsistent data.
4. Prepare or verify bounding box annotations  
   Ensure that each image has correct bounding box labels for localization.
5. Split the dataset  
   Divide the data into training, validation, and test sets.
6. Select candidate models  
   - Classical ML approach: potentially SVM  
   - Deep learning approach: potentially AlexNet or VGG16
7. Train the model  
   Train selected models on the prepared dataset.
8. Evaluate performance  
   Compare model results using suitable evaluation metrics.
9. Analyze results and refine the pipeline  
   Identify weaknesses and improve preprocessing, model choice, or training strategy.

### 5.3 Model Discussion
We are considering both classical and deep learning approaches. A classical ML model such as HOG features combined with SVM may be used as a baseline. We are also considering deep learning models such as AlexNet and VGG16, with modifications of their heads to predict bounding boxes.

## 6. Draft Concept Map

_(To be added.)_
