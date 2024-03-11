
import React, { useState, useEffect} from 'react';

 

  function VideoViewer({data}) {

    const [srcUrl, setSrcUrl] = useState('')


    useEffect(
        () => {
            if (data['Path'] ) {
                
              setSrcUrl(`http://localhost:8988/playvideo/?path=${data['Path']}`)
            }else{
                const url = URL.createObjectURL(data);
                setSrcUrl(url);
                
            }
        },
        [data]
    );
  

    return (
        <>
    <video src={srcUrl} controls muted></video>
         </>
 
    );
  }
  export default VideoViewer;
  