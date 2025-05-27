# aquaDetect-backend
Backend using flask for AquaDetect : Fish Disease Detection

How to run?

`git clone https://github.com/nik410/aquaDetect-backend.git`

### Open aquadetect-backend project in pycharm

##### create a .venv
`Settings-> Project:aquadetect-backend-> Python Interpreter-> Add interpreter-> Add local interpreter`  

###### Ensure following details are met

`Environment: Generate new`  
`Type: virtual env`  
`An appropirate base python is selected`  
`Location :  /Users/knewatia/Desktop/p/Pycharm/aquaDetect-backend/.venv`  NOTE: .venv MUST be created inside the cloned project  

Once .venv has been created make sure u are inside the directory and run

`pip install opencv-python numpy tensorflow scikit-learn matplotlib seaborn scikit-image Flask Flask-Cors`  

after relevant libraries has been installed download the 2 models from whatsApp to `models/` folder  

then run `python backend.py`

Test the api using 

`curl --request POST \
  --url http://127.0.0.1:5000/uploadimg \
  --header 'content-type: multipart/form-data' \
  --form image=@/Users/knewatia/Desktop/p/Pycharm/aquaDetect-backend/testimage.jpeg`
