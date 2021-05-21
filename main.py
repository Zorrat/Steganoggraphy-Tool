####### CORE IMPORTS ###############

import streamlit as st
from PIL import Image,ImageEnhance
import numpy as np
import os
import sys
import wave
import hashlib
import math
from stegano import lsb
import numpy as np
import cv2
import shutil
import time
import hashlib


###################### VideoSteg Helper Functions #############################
def clrTemp(path = './vidTemp'):
    '''
    Parameters
    ----------
    path : string, optional
        DESCRIPTION. The default is './vidTemp'.

    Returns
    -------
    None.

    '''
    if os.path.exists(path):
        shutil.rmtree(path)
        print("[INFO] tmp files are cleaned up")

def getSha2(filename):
    '''
    Calculates the SHA256 Hash of file in the file system

    Parameters
    ----------
    filename : string
        DESCRIPTION.
            Path of the file in OS or relative path.
    Returns
    -------
    String: Hex Representation of the sha256 hash

    '''
    sha256_hash = hashlib.sha256()
    with open(filename,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
        return(sha256_hash.hexdigest())

def SplitString(string,n):
    '''
    Parameters
    ----------
    string : str
        DESCRIPTION Message to split
    n : int
        DESCRIPTION: Difficulty value

    Returns
    -------
    splitStr : list
        DESCRIPTION: string evenly split into segments.

    '''
    if n > len(string):
        return("Message Too Small Choose Lower Difficulty")
    
    splitStr = np.array_split(list(string),n)
    splitStr = ["".join(i) for i in splitStr]
    return (splitStr)

def FrameExtract(video):
    '''
    Extracts the frames to a temp folder
    Parameters
    ----------
    video : str
        DESCRIPTION: Path to the video file

    Returns
    -------
    count : Total Number of Frames
    fps : framerate
        framerate of the input video

    '''
    if not os.path.exists("./vidTemp"):
        os.makedirs("vidTemp")

    temp_folder="./vidTemp"
    print("[INFO] tmp directory is created")

    vidcap = cv2.VideoCapture(video)
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("fps= ",fps)

    while vidcap.isOpened():
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(temp_folder, "{:d}.png".format(count)), image)
        count += 1
    vidcap.release()
    return count,fps

def getFrame2Message(message,frameCount,sha,difficulty = 4):
    """
    Generates a Dictionary mapping message segments to frames of the video

    Parameters
    ----------
    message : str
        Message to Hide
    frameCount : int
        Framerate of the video
    sha : str
        SHA256 hex string.
    difficulty : int, optional
        this value decides how many segments the message is split into. The default is 4.

    Returns
    -------
    Dictionary

    """
    frame2message = {}
    segments = 64//difficulty 
    frames = getFrameOrder(difficulty,frameCount,sha)
    strSplit = SplitString(message,segments)
    print("Segmented message:",strSplit)
    for i,j in zip(frames,strSplit):
        frame2message[i] = j
    return(frame2message)


def encodeString(frame2message,root="./vidTemp/"):
    """
    Encode the message to the frames in temp folder with lsb steg

    Parameters
    ----------
    frame2message : Dict
    root : str, optional
        path to temp folder The default is "./vidTemp/".

    """
    for i,splitString in frame2message.items():
        f_name="{}{}.png".format(root,i)
        secret_enc=lsb.hide(f_name,splitString)
        secret_enc.save(f_name)
        print("[INFO] frame {} holds {}".format(f_name,splitString))

def decodeString(video,difficulty = 4):
    """
    lsb decode the frames from the temp folder and return the full hidden message as a string

    Parameters
    ----------
    video : str
        Path to the encoded video.
    difficulty : int, optional
        this value decides how many segments the message is split into. The default is 4.

    Returns
    -------
    str : Hidden Message

    """
    frameCount,fps = FrameExtract(video)
    print("Decoding FPS:",fps)
    secret=[]
    os.system("ffmpeg -i "+video+" -q:a 0 -map a vidTemp/audio.mp3 -y")
    sha = hashlib.sha256(str(frameCount+fps).encode('utf-8')).hexdigest()
    print("Decoding Sha:",sha)
    root="./vidTemp/"
    frames = getFrameOrder(difficulty,frameCount,sha)
    print("Frames: ",frames)
    for i in frames:
        f_name="{}{}.png".format(root,i)
        print("FileName:",f_name)
        secret_dec=lsb.reveal(f_name)
        if secret_dec == None:
            continue
        secret.append(secret_dec)
    
    print("Secret:",secret)
    clrTemp()
    return(''.join([i for i in secret]))
            

def getFrameOrder(difficulty,frameCount,sha):
    """ 
    Internal Function
    """
    frames = []
    for i in range(0,64,difficulty):
        frameNo = int(sha[i:i+difficulty],16) % frameCount
        frames.append(frameNo)
    return (frames)


def EncodeVid(path,message,difficulty = 8):
    """
    Wrapper function for putting the entire encoding process together

    Parameters
    ----------
    path : str
        Path to cover base video file.
    message : str
        Message to Hide.
    difficulty : int, optional
        this value decides how many segments the message is split into. The default is 8.

    Returns
    -------
    None

    """
    print ("Processing, Please Wait")
    frameCount,fps = FrameExtract (path)
    print("Extracting Audio")
    os.system("ffmpeg -i "+path+" -q:a 0 -map a vidTemp/audio.mp3 -y")
    sha = hashlib.sha256(str(frameCount+fps).encode('utf-8')).hexdigest()
    print("SHA256 of Encoding Video:",sha)
    frame2message = getFrame2Message(message,frameCount,sha,difficulty)
    for k,v in frame2message.items():
        print("Frame Number: "+str(k)+" holds : "+str(v))
    encodeString(frame2message)
    os.system("ffmpeg -framerate "+str(int(fps))+" -i vidTemp/%d.png -vcodec libx264rgb -crf 0 vidTemp/video.avi -y")
    if "audio.mp3" in os.listdir("./vidTemp"):
        os.system("ffmpeg -i vidTemp/video.avi -i vidTemp/audio.mp3 -vcodec libx264rgb -crf 0 EncodedVideos/EncVideo"+message[0:2]+".avi -y") 
    else:
        os.system("ffmpeg -i vidTemp/video.avi -vcodec libx264rgb -crf 0 EncodedVideos/EncVideo"+message[0:2]+".avi -y")
    print("Video Encoded")
    #clrTemp()
    
def DecodeVid(path,difficulty = 4):
    """ 
    Internal Wrapper Function
    """
    return decodeString(path,difficulty)

############################

#Helper Functions

def bgr2rgb(img):
    """ 
    Internal Function for OpenCV
    """
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return im_rgb

def file_selector(folder_path='.'):
    """ 
    Internal Function for Selecting Files in OS
    """
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    path = os.path.join(folder_path, selected_filename)
    return  path
def loadim(img_file):
    """ 
    Internal Function to load image to memory
    """
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img

def EncodeIm(src, message):
    """
    Encodes image with LSB steganography

    Parameters
    ----------
    src : str
        Path to input image.
    message : str
        Message to hide in the image

    """

    img = Image.open(src, 'r')
    width, height = img.size
    array = np.array(list(img.getdata()))

    if img.mode == 'RGB':
        n = 3
        m = 0
    elif img.mode == 'RGBA':
        n = 4
        m = 1

    total_pixels = array.size//n

    message += "gpXIII"
    b_message = ''.join([format(ord(i), "08b") for i in message])
    req_pixels = len(b_message)

    if req_pixels > total_pixels:
        print("ERROR: Need larger file size")

    else:
        index=0
        for p in range(total_pixels):
            for q in range(m, n):
                if index < req_pixels:
                    array[p][q] = int(bin(array[p][q])[2:9] + b_message[index], 2)
                    index += 1

        array=array.reshape(height, width, n)
        enc_img = Image.fromarray(array.astype('uint8'), img.mode)
        enc_img.save("EncodedImages/Encoded_"+message[0]+".png")
        print("Image Encoded Successfully")
        
        
def DecodeIm(src):
    """
    Decode a Image which has stego message embeded in it

    Parameters
    ----------
    src : str
        Path to image.

    Returns
    -------
    str
        Decoded Message

    """

    img = Image.open(src, 'r')
    array = np.array(list(img.getdata()))

    if img.mode == 'RGB':
        n = 3
        m = 0
    elif img.mode == 'RGBA':
        n = 4
        m = 1

    total_pixels = array.size//n

    hidden_bits = ""
    for p in range(total_pixels):
        for q in range(m, n):
            hidden_bits += (bin(array[p][q])[2:][-1])

    hidden_bits = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]

    message = ""
    for i in range(len(hidden_bits)):
        if message[-6:] == "gpXIII":
            break
        else:
            message += chr(int(hidden_bits[i], 2))
    if "gpXIII" in message:
        return(message[:-6])
    else:
        return("No Hidden Message Found")


def EncodeAu(path,message):
    """
    Function to encode Audio Files with Steganography Message

    Parameters
    ----------
    path : str
        Path to Audio file to encode.
    message : str
        Message to Hide.

    """

    # read wave audio file
    song = wave.open(path, mode='rb')
    # Read frames and convert to byte array
    frame_bytes = bytearray(list(song.readframes(song.getnframes())))

    # The "secret" text message
    string=message
    # Append dummy data to fill out rest of the bytes. Receiver shall detect and remove these characters.
    string = string + int((len(frame_bytes)-(len(string)*8*8))/8) *'#'
    # Convert text to bit array
    bits = list(map(int, ''.join([bin(ord(i)).lstrip('0b').rjust(8,'0') for i in string])))

    # Replace LSB of each byte of the audio data by one bit from the text bit array
    for i, bit in enumerate(bits):
        frame_bytes[i] = (frame_bytes[i] & 254) | bit
    # Get the modified bytes
    frame_modified = bytes(frame_bytes)

    # Write bytes to a new wave audio file
    with wave.open('EncodedAudio/audio_'+message[0]+'.wav', 'wb') as fd:
        fd.setparams(song.getparams())
        fd.writeframes(frame_modified)
    song.close()
    #return('audio_'+message[:5]+'.wav')
    
def DecodeAu(path):
    """
    Decode a Stego encoded audio file 

    Parameters
    ----------
    path : str
        Path to audio file.

    Returns
    -------
    decoded : str
        Decode message.

    """
    song = wave.open(path, mode='rb')
    # Convert audio to byte array
    frame_bytes = bytearray(list(song.readframes(song.getnframes())))

    # Extract the LSB of each byte
    extracted = [frame_bytes[i] & 1 for i in range(len(frame_bytes))]
    # Convert byte array back to string
    string = "".join(chr(int("".join(map(str,extracted[i:i+8])),2)) for i in range(0,len(extracted),8))
    # Cut off at the filler characters
    decoded = string.split("###")[0]

    # Print the extracted text
    #print("Sucessfully decoded: "+decoded)
    song.close()
    return decoded
    




###### Main ############

def main():
    """
    Main Function to run the Streamlit UI

    Returns
    -------
    None.

    """
    st.title("Steganography Tool")
    st.subheader("App with various Steganography tools \nMade by Group 13 (prototype build)")
    
    
    # Sidebar activites defined
    st.sidebar.subheader("Select which Editing Function you would like to Use")
    activities = ['About','Image',"Audio","Video"]
    sidebar_choice = st.sidebar.selectbox('Select file Type',activities)
    
    if sidebar_choice == 'About':
        st.subheader("This tool was developed as a final Year project for Engineering Degree at Xavier Institute of Engineering.\n This application includes Applying steganography to Images and Audio files. \n A New implementation of Psuedo-random Steganography for video files is also implemented.")
    
    if sidebar_choice == "Image":
        st.subheader("Select to either Encode or Decode Image")
        operationsIm = ['Encode','Decode']
        operationIm = st.selectbox("Select Operation",operationsIm)
        if operationIm == "Encode":
            st.info("Add Image and Private message")
            message = st.text_input("Add Your Private Message","Lorem Ipsum is simply dummy text of the printing and typesetting industry.")
              #Uploading Main Image

            path = file_selector("./Input")
            if st.button("Apply and Save") and (path is not None) :
                EncodeIm(path,message)
                st.image(Image.open(path, 'r'),use_column_width = True)
                st.success("Image Successfully Encoded and saved")
            else:
                st.info("Please Select an Image")
        elif operationIm == "Decode":
            st.info("Select Image to decode from")
  
            path = file_selector("./EncodedImages")
           #print(path)
            if st.button("Apply and Save") and (path is not None) :
                mes = DecodeIm(path)
                st.success("Image Successfully Decoded")
                #st.image(Image.open(path, 'r'))
                st.info("Decoded Message: " + mes)
            else:
                st.info("Please Select an Image")
    
    elif sidebar_choice == "Audio":
        st.subheader("Select to either Encode or Decode Audio")
        operationsAu = ['Encode','Decode']
        operationAu = st.selectbox("Select Operation",operationsAu)
        if operationAu == "Encode":
            st.info("Add Audio wave file and a Private message")
            message = None
            message = st.text_input("Add Your Private Message","Lorem Ipsum is simply dummy text of the printing and typesetting industry.")
            path = file_selector("./Input")
            if st.button("Apply and Save") and (path is not None) :
                EncodeAu(path,message)
                st.audio(path,format = 'audio/wav')
                st.success("Audio Successfully Encoded and saved")
            else:
                st.info("Please Select a audio File")
        elif operationAu == "Decode":
            st.info("Select Audio file to decode from")

            path = file_selector("./EncodedAudio")
            print(path)
            if st.button("Apply and Save") and (path is not None) :
                mes = DecodeAu(path)
                st.success("Audio Successfully Decoded")
                st.info("Decoded Message " + mes)
            else:
                st.info("Please Select a Audio file")
      

    elif sidebar_choice == "Video":
        st.subheader("Select to either Encode or Decode Audio")
        operationsVi = ['Encode','Decode']
        operationVi = st.selectbox("Select Operation",operationsVi)
        if operationVi == "Encode":
            st.info("Add a Video File file and a Private message and choose a difficulty")
            message = None
            message = st.text_input("Add Your Private Message","Lorem Ipsum is simply dummy text of the printing and typesetting industry.")
            path = file_selector("./Input")
            difficulty = st.slider("Select Difficulty Optional",min_value = 1,max_value = 4,value = 3)
            difficulty_map = {1:16,2:8,3:4,4:2}
            difficulty = difficulty_map[difficulty]
            if st.button("Apply and Save") and (path is not None) :
                EncodeVid(path,message,difficulty)
                st.success("Video Successfully Encoded and saved")
            else:
                st.info("Please Select a Video File")
        elif operationVi == "Decode":
            st.info("Select Video file to decode from")

            path = file_selector("./EncodedVideos")
            difficulty = st.slider("Select Difficulty Optional",min_value = 1,max_value = 4,value = 3)
            difficulty_map = {1:16,2:8,3:4,4:2}
            difficulty = difficulty_map[difficulty]
            if st.button("Apply") and (path is not None):
                mes = DecodeVid(path,difficulty)
                st.success("Audio Successfully Decoded")
                st.info("Decoded Message: " + mes)
            else:
                st.info("Please Select a Video file")







if __name__ == '__main__':
    main()