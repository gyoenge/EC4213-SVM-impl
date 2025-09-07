# Gist Machine Learning Course Coding Assignment
## SVM (Support Vector Machine) Implementation

- Course: GIST Machine Learning (EC4213)
- Project Type: SVM Implementation Individual Coding Assignment 

### Description
An educational project focusing on the **implementation of Support Vector Machines (SVM) from scratch**. 
The project covers key concepts such as **hard margin SVM, soft margin SVM (primal and dual formulations), and the use of kernel tricks to handle non-linear data**.

- `SVM_hard.py` : Implementation of SVM with hard margin.
- `SVM_soft.py` : Implementation of SVM with soft margin.
- `SVM_kernel.py` : Implementation of kernels which will be used to soft margin.
- `utils.py` : A bunch of utility functions!
- `test.py` : A testing code! We run this code to evaluate implementation.

### Overview 

#### Hard margin SVM

<p align="center">
<img width="50%" alt="HSVM" src="https://github.com/user-attachments/assets/f82fc6ca-febe-4af0-b48f-69686a3d9eb0" />
</p>  

- We implemented the process of finding the optimal decision boundary using **hinge loss** and **coordinate gradient descent**.

#### Soft margin SVM 

<p align="center">
<img width="50%" alt="SSVM1" src="https://github.com/user-attachments/assets/d77da26f-df0a-4502-817b-031311ecbf22" />
</p>  
<p align="center">
<img width="50%" alt="SSVM2" src="https://github.com/user-attachments/assets/83738b1a-5dc2-4126-b03c-2ec418b15f65" />
</p>  

- We can find a decision boundary of two classes by solving dual problem. Slack variables allow misclassification.

#### Kernel Tricks 

<p align="center">
<img width="50%" alt="KSVM" src="https://github.com/user-attachments/assets/04392223-fb9e-4937-8757-cd6791424232" />
</p>

- We implemented various kernel filters to SVM to compare their performance. 

