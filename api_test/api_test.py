from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from predict import *
import databases
import sqlalchemy
from sqlalchemy.sql import text
from minio import Minio
from minio.error import S3Error
import datetime
import urllib
import shutil
import json
import glob
import re
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware)

host_server = os.environ.get('host_server', 'dumbo.db.elephantsql.com')
db_server_port = urllib.parse.quote_plus(str(os.environ.get('db_server_port', '5432')))
database_name = os.environ.get('database_name', '')
db_username = urllib.parse.quote_plus(str(os.environ.get('db_username', '')))
db_password = urllib.parse.quote_plus(str(os.environ.get('db_password', '')))
ssl_mode = urllib.parse.quote_plus(str(os.environ.get('ssl_mode','prefer')))
DATABASE_URL = 'postgresql://{}:{}@{}:{}/{}?sslmode={}'.format(db_username, db_password, host_server, db_server_port, database_name, ssl_mode)
database = databases.Database(DATABASE_URL)

engine = sqlalchemy.create_engine(
    #DATABASE_URL, connect_args={"check_same_thread": False}
    DATABASE_URL, pool_size=3, max_overflow=0
)




# bucket access
bucket_access_key_id = ''
bucket_secret_access_key = ''


# bucket name
bucket_name = 'deepfake-detection'


client = Minio("play.min.io",
    access_key=bucket_access_key_id,
    secret_key=bucket_secret_access_key,
)

found = client.bucket_exists(bucket_name)
if not found:
    client.make_bucket(bucket_name)
    print("\n")
    print("Created bucket", bucket_name)
    print("\n")
else:
    print("\n")
    print("Bucket", bucket_name, "already exists")
    print("\n")





#table info
table_name = '"public"."videos-table"'
table_cols = '(vidname,path,email,predicted_output,imagepaths)'

# db queries
insert_new = """INSERT INTO {} {} VALUES ( '{}', '{}','{}','{}', '{}');"""
delete_row = """DELETE FROM {} WHERE path = '{}';"""
email_rows = """SELECT * FROM {} WHERE email = '{}';"""
image_paths = """SELECT imagepaths FROM {} WHERE path = '{}';"""


# Regex for email validate
regex_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'

# Directory where uploaded videos will be saved
UPLOAD_DIRECTORY = "temp"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


def upload_bucket_db(vidname="", path="", temp_path=""):
    try:
    
        # Upload the image files, renaming it in the process
        client.fput_object(
                bucket_name, path, os.path.join(temp_path, vidname)
            )
        
        print("successfully uploaded the video")      
        print("\n") 
                    
        os.remove(os.path.join(temp_path, vidname))
        for i in glob.glob(temp_path + '/*.png'):
            os.remove(i)
        os.rmdir(temp_path)

    except Exception as e:
        print(e)



@app.post("/upload/")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...), email: str = Form(...)):
   
    try:
        temp_path = os.path.join(UPLOAD_DIRECTORY, video.filename)
        os.makedirs(temp_path)
        temp_vidpath = os.path.join(temp_path,video.filename)

        # Save the uploaded video to the specified directory
        with open(temp_vidpath, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Call function to perform inference on the uploaded video
        try:
            result = predict(vid_path=temp_vidpath, temp_path=temp_path)
        except Exception as e:
            print(e)

        if not (re.fullmatch(regex_email, email)):
            email = 'AnonymousUser'
        
        if result:
            temp_result = str(result).replace("'","")
        else:
            temp_result = ""


        try:
            temp_filenames = []
            ct = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{ct}_{video.filename}"
            found = client.bucket_exists(bucket_name)
            if not found:
                client.make_bucket(bucket_name)

            for i in glob.glob(temp_path + "/*.png"):

                temp_name = i.split("/")[-1]
                temp_filenames.append(f"{ct}_{temp_name}")

                # Upload the image files, renaming it in the process
                client.fput_object(
                    bucket_name, f"{ct}_{temp_name}", i
                )


            result["Image Paths"] = temp_filenames


            temp_filenames = str(temp_filenames).replace("'","").replace('"','')

            print("successfully uploaded the files")      
            print("\n") 
            try:
                query_result = engine.execute(text(insert_new.format(table_name, table_cols, video.filename,path, email, temp_result, temp_filenames)))
            except Exception as e:
                print(e)

            background_tasks.add_task(upload_bucket_db, vidname=video.filename, path=path, temp_path=temp_path)
        except Exception as e:
            print("\n")
            print("error occurred.", e)
            print("\n")
        
        result["Image Paths"] = temp_filenames
        # Return the result of the inference
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   



@app.post("/delete/")
async def delete_video(path: str):
    try:
        query_result = engine.execute(text(delete_row.format(table_name, path)))
        # print(query_result.rowcount)
        return({'isDeleted': True if query_result.rowcount else False })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/videos/")
async def list_videos(email: str):
    try:
        query_result = engine.execute(text(email_rows.format(table_name, email))).all()
        result_response = {}
        if query_result:
            for i in query_result:
                result_response.update({i[1]:i[0]})
        return result_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{image_name}")
async def get_image(image_name: str):
    try:
            
        response = client.get_object(bucket_name, image_name)
        headers = {
            
            "Access-Control-Allow-Origin": "*" ,
            "Access-Control-Allow-Methods": "POST, GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Content-Type": "image/png",
  
        }
        
        return Response(content=response.read(), headers=headers, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/playvideo/")
async def get_video(path: str):
    try:
        response = client.get_object(bucket_name, path)
        headers = {
            
            "Access-Control-Allow-Origin": "*" ,
            "Access-Control-Allow-Methods": "POST, GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Content-Type": "video/mp4",
  
        }
        
        return Response(content=response.read(), headers=headers, media_type="video/mp4")

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error fetching video: {err}")





@app.post("/predict/")
async def predict_video(path: str):
    try:
        # Specify the local path where you want to save the downloaded file
        temp_vidpath = os.path.join(UPLOAD_DIRECTORY, path)

        try:
            # Download the file from MinIO
            client.fget_object(bucket_name, path, temp_vidpath)
        
            result = predict(vid_path=temp_vidpath)
            result['Path'] = result['Video Name']
            result['Video Name'] = '_'.join(result['Video Name'].split("_")[2:])
            result['Video Name'] = '_'.join(result['Video Name'].split("_")[2:])
            query_result = engine.execute(text(image_paths.format(table_name, path))).all()[0]["imagepaths"]
            result["Image Paths"] = query_result



        except Exception as e:
            print(e)

  


        os.remove(temp_vidpath)

        return result
        
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))