# Baleegh Backend
Welcome! This repository contains the backend code for the Baleegh project.

## Technologies Used
* **FastAPI Framework**: For creating our APIs.
* **Modal Hosting**: For hosting the backend services.
* **IBM watsonx**: Used for interacting with the ALLaM model.
* **Hugging Face**: For downloading the model from Hugging Face's platform.
## Project Structure
### `api` Folder
Contains all components related to the APIs.

### `service` Folder
Houses the business logic of the backend.

### `util` Folder
Includes utility functions and foundational modules for interacting with the ALLaM model.

## Backend Architecture
![image](https://github.com/user-attachments/assets/193fb3d0-a2e9-465d-8d41-4ccf7e732dcc)

## Backend Performance
### Running on the cpu:

**Warm/ready containers:** 3 containers
<br/>
**Queue Average Time:** 45 milliseconds
<br/>
**Startup Average Time:** 3 min/cold container
<br/>
**Excute Average time:** 4 seconds

### Running on the gpu:

**Warm/ready containers:** 3 containers
<br/>
**Queue Average Time:** 25 milliseconds
<br/>
**Startup Average Time:** 3 min/cold container
<br/>
**Excute Average time:** 2 seconds

**_NOTE:_** this meatures is not accurate 100%

