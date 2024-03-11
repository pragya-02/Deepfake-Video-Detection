# deepfake-detection
The DeepFake Detection Web Application is a web application designed to identify whether uploaded videos are authentic or deepfake. 

The detection model is based on CNN with LSTM architecture that scrutinizes the temporal facial movement characteristics within videos to determine their authenticity. The model is trained on the preprocessed dataset. The dataset is preprocessed by resizing, cropping face area and extracting landmarks, blendshapes and facial crops from the cropped face. Further, the model is tested to ensure its effectiveness and performance.

The web application offers a user-friendly interface, built using ReactJS. It connects seamlessly with the backend inference server through a FastAPI-based Python API. Moreover, the web app features account sign-in and sign-up functionalities, facilitating convenient access to user-uploaded videos.

For the further information on the project like dataset, model performance etc., please refer the documentation provided in the code section.
For the access to the dataset and model weights, please contact us at [Deepfakedetectionprojectkhwopa@gmail.com](mailto:deepfakedetectionprojectkhwopa@gmail.com).

## Installation
A. Train Jupyter Notebook and API Source Code:
 - Can be opened and edited in the code editor like VS code.

 The API server can be started by running the command.
```bash
uvicorn api_test:app
```
     
B. Frontend ReactJS Source Code:
- Can be opened and edited in the code editor like VS code.

The development server can be started by running the command.
```bash
npm start
```


## Usage

The web app can be browsed in the browser after running the server.

## Youtube Link Showing the Demo
[DeepFake Detection Webapp's Demo](https://www.youtube.com/embed/Tgo_5fGszJQ)  


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
