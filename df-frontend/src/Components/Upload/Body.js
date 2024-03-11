import React, { useState, useEffect } from "react";
import BodyHeader from "./BodyHeader";
import BodyUploader from "./BodyUploader";
import { CircleLoader } from "react-spinners";
import axios from "axios";
import { LuFileVideo } from "react-icons/lu";
import { FaTrash } from "react-icons/fa6";

const Body = (props) => {
  const [data, setData] = useState([]);
  const [isLoadingRem, setIsLoadingRem] = useState(true);
  const [loadFromHistoryVideos, setLoadFromHistoryVideos] = useState(false);
  const [isLoadingFromHistoryVideo, setLoadingFromHistoryVideo] = useState(false);






  const reload_data = () => {
    if (props.profile) {
      setIsLoadingRem(true);
      axios.post(
          `http://localhost:8988/videos/?email=${props.profile.email}`,
          {
            headers: {
              "ngrok-skip-browser-warning": "0",
            },
          }
        )
        .then((res) => {
          const arr = [];
          Object.keys(res.data).forEach((key) =>
            arr.push({ path: key, vidname: res.data[key] })
          );
          setData(arr);
        })
        .catch((err) => console.log(err))
        .finally(setIsLoadingRem(false));
    }
  };

  const predictVideo = async (path) => {
    try {
      setLoadingFromHistoryVideo(true);
      const response = await axios.post(
        `http://localhost:8988/predict/?path=${path}`,
        {
          headers: {
            "ngrok-skip-browser-warning": "0",
          },
        }
      );
      const temp_response = response.data
      
      temp_response["Image Paths"] = temp_response["Image Paths"]
      .slice(1, -1)
      .split(',')
      .map(path => path.trim())

      setLoadFromHistoryVideos(temp_response);
     
      
    } catch (error) {
      console.error(error);
    } finally {
      setLoadingFromHistoryVideo(false);
    }
  };

  const delVideo = async (path) => {
    try {
      setIsLoadingRem(true);
      const response = await axios.post(
        `http://localhost:8988/delete/?path=${path}`,
        {
          headers: {
            "ngrok-skip-browser-warning": "0",
          },
        }
      );

      if (response.data.isDeleted) {
        setData(data.filter((video) => video.path !== path));
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoadingRem(false);
    }
  };



  useEffect(() => {
    if (props.profile) {
      setIsLoadingRem(true);
      const getVideoListData = async () => {
      
      await axios.post(
          `http://localhost:8988/videos/?email=${props.profile.email}`,
          {
            headers: {
              "ngrok-skip-browser-warning": "0",
            },
          }
        )
        .then((res) => {
          const arr = [];
          Object.keys(res.data).forEach((key) =>
            arr.push({ path: key, vidname: res.data[key] })
          );
          setData(arr);
        })
        .catch((err) => console.log(err))
        .finally(setIsLoadingRem(false));
      }
      getVideoListData();
    }
  }, [props.profile]);

  const historyList = isLoadingRem ? (
    <div className="loader-container flex items-center justify-center pt-8 pb-4">
      <CircleLoader animation="border" color="#FFFFFF" />
    </div>
  ) : data.length > 0 ? (
    data.map((video) => (
      <div className="flex gap-4 items-center">
        <button
          type="submit"
          buttonName="Scan"
          className="bg-transparent w-5/6 flex items-center text-text hover:text-primary font-heading py-2 px-4 border-[2px] border-border hover:border-transparent rounded-lg"
          onClick={() => predictVideo(video.path)}
        >
          <LuFileVideo />
          &nbsp; {video.vidname}
        </button>
        <button onClick={() => delVideo(video.path)}>
          <FaTrash className="hover:text-primary" />
        </button>
      </div>
    ))
  ) : (
    <div className="flex gap-4 items-center">
      <p className="font-body text-[0.7rem] opacity-50 pt-2">
        <i>Nothing to show</i>
      </p>
    </div>
  );

  return (
    <>
      <div className="bg-background p-8 h-screen w-full flex flex-col">
        <BodyHeader />
{isLoadingFromHistoryVideo ? <div className="loader-container flex items-center justify-center pt-16 pb-8">
              <CircleLoader animation="border" color="#FFFFFF" />
            </div> : <BodyUploader
          email={props.profile ? props.profile.email : "AnonymousUser"}
          reload={reload_data}
          loadFromHistoryVideos={loadFromHistoryVideos}
        />}
        
      </div>
      {props.profile ? (
        <div className="absolute w-80 top-[82px] right-[0px] border-[2px] border-border p-4 font-body text-text rounded-bl-lg">
          <h1 className="border-b-[2px] border-border w-full text-[1.2rem]">
            History
          </h1>
          <div className="flex flex-col py-4 gap-2 max-h-80 overflow-scroll ">
            {historyList}
          </div>
        </div>
      ) : (
        ""
      )}
    </>
  );
};

export default Body;
