import classes from "./BodyUploader.module.css";
import "./Modal.css";
import { MdFileUpload } from "react-icons/md";
import { LuScanFace } from "react-icons/lu";
import { BiReset } from "react-icons/bi";
import { MdOpenInBrowser } from "react-icons/md";
import { useState} from "react";
import { CircleLoader } from "react-spinners";
import axios from 'axios';
import VideoViewer from "./VideoViewer";
import { IoClose } from "react-icons/io5";


const BodyUploader = ({ email, reload, loadFromHistoryVideos }) => {
  const [file, setFile] = useState(false);
  const [responseData, setresponseData] = useState(loadFromHistoryVideos);
  const [isLoading, setIsLoading] = useState(false);
  const [cancelToken, setCancelToken] = useState(null);


  const handleFile = (event) => {
    setFile(event.target.files[0]);
    setresponseData(false);
  };

  const resetHandler = () => {
    if (cancelToken) {
      cancelToken.cancel('Request canceled by user');
    }
    setFile(false);
    setIsLoading(false);
    setresponseData(false);
  };

  

  const fileUploadHandler = async (event) => {
    event.preventDefault();
    if (file){
          setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("video", file, file.name);
      formData.append("email", email);

      const source = axios.CancelToken.source();
      setCancelToken(source);
      const response = await axios.post(
        "http://localhost:8988/upload/",
       
        formData,
        {
          cancelToken: source.token,
          headers: {
            "Content-Type": "multipart/form-data",
            "ngrok-skip-browser-warning": "0",
          },
        }
      );
        
      const temp_response = response.data
      
      temp_response["Image Paths"] = temp_response["Image Paths"]
      .slice(1, -1)
      .split(',')
      .map(path => path.trim())

      setresponseData(temp_response)
  

      reload();
   
    } catch (error) {
      if (axios.isCancel(error)) {
        console.log('Request canceled', error.message);
        setresponseData(false);
      }else {
        console.error(error);
      }
      
    } finally {
      setIsLoading(false);

    }
  }
  };

  let fileName = responseData ? responseData['Video Name'] : '';
  if (file) {
    fileName = file.name
  }

  const [isOpen, setIsOpen] = useState(false);

  const toggleModal = () => {
    setIsOpen(!isOpen);
    setIsLoading(false);


  };




  return (
    <>
      {isOpen ? <div className="modal-overlay">
              <div className="modal">
                <div className="modal-header">
                  <h2 className="font-heading text-[1.4rem]">More Details</h2>
                  <button className="close-button" onClick={toggleModal}>
                  <IoClose size={28} />
                  </button>
                </div>
                <div className="modal-content">
                  <div className="flex flex-col px-8 py-4 gap-16 max-h-[25rem] overflow-y-scroll">
                  <div className="flex flex-col gap-6 items-center justify-center">
                  <p>Cropped Face after Detection and Alignment</p>
                  <img src={`http://localhost:8988/images/${responseData["Image Paths"][3]}`} alt=""/>
                  </div>
                  <div className="flex flex-col gap-4 items-center justify-center">
                  <p>Landmarks</p>
                  <img src={`http://localhost:8988/images/${responseData["Image Paths"][2]}`} alt=""/>
                  </div>
                  <div className="flex flex-col gap-4 items-center justify-center">
                  <p>Blendshape Scores</p>
                  <img src={`http://localhost:8988/images/${responseData["Image Paths"][1]}`} alt=""/>
                  </div>
                  <div className="flex flex-col gap-4 items-center justify-center">
                  <p>Cropped Facial Parts</p>
                  <img src={`http://localhost:8988/images/${responseData["Image Paths"][0]}`} alt=""/>
                  </div>
                  </div>
                  
                  
                </div>
              </div>
            </div> : ""
          }

      <div className={classes["body-holder"]}>
       <div className={classes["video-holder"]}>
        {file ? <VideoViewer data={file} /> :  responseData ? <VideoViewer data={responseData}/> : "" } 
       </div>
        <form onSubmit={fileUploadHandler} className={classes["body-form"]}>
          <div className={classes["upload-holder"]}>
            <label>
              <div className={classes["form-label"]}>
                <div className="flex items-center justify-center gap-2">
                  <h1>
                    {!fileName
                      ? "Click here to upload the video"
                      : fileName}
                  </h1>
                  <MdFileUpload size={22} />
                </div>
              </div>
              <input type="file" onChange={handleFile} accept="video/*"></input>
            </label>
          </div>
          {isLoading ? (
            <div className="loader-container flex items-center justify-center pt-16 pb-8">
              <CircleLoader animation="border" color="#FFFFFF" />
            </div>
          ) : !responseData ? "" : (
            <div className="flex flex-col pt-8 items-center font-body justify-center gap-8 text-text">
              <div className="flex flex-col gap-2 items-center font-body justify-center">
                <h1>
                  Predicted Label: <b>{responseData["Predicted Output"]}</b>
                </h1>
                <h1>
                  Confidence of Prediction:{" "}
                  <b>{responseData["Prediction Confidence"]}</b>
                </h1>
                <h1>
                  Inference Time: <b>{responseData["Inference Time"]}</b>
                </h1>
              </div>
              <button
                type="button"
                onClick={toggleModal}
                className="flex items-center justify-center text-primary hover:text-accent"
              >
                <MdOpenInBrowser size={20} /> &nbsp; More Details
              </button>
            </div>
          )}

          <div className={classes["button-holder"]}>
            <button
              type="submit"
              onClick={fileUploadHandler}
              buttonName="Scan"
              className="bg-primary flex items-center text-text hover:drop-shadow-[0_35px_35px_rgba(58,49,216,0.5)] font-heading hover:text-text mx-8 py-2 px-4 border-[2px] border-border hover:border-transparent rounded-lg disabled:opacity-50"
              disabled={isLoading || !file || responseData ? true : false}
              // disabled={true}
            >
              <LuScanFace />
              &nbsp; Scan
            </button>
            <button
              type="reset"
              onClick={resetHandler}
              buttonName="Reset"
              className="bg-transparent flex items-center text-text hover:text-primary font-heading  mx-8 py-2 px-4 border-[2px] border-border hover:border-transparent rounded-lg"
            >
              <BiReset />
              &nbsp; Reset
            </button>
          </div>
        </form>

        {/* <h1 className="text-text font-body">
                  <b>Currently, The Remote Inferencing Server is Offline.</b>
                </h1> */}
                
      </div>
    </>
  );
};

export default BodyUploader;
