from extract_data_profile import * 
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post('/uploadfile/')
async def upload_file(file: UploadFile = File(...)):
    print('start method')

    global data 
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file.file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file.file)
    else: 
        return {'message': 'choose right file type'}

    profiler = DatasetProfiler(data)
    profile = profiler.build_profile()
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    
    return {'detail': 'file uploaded successfully'}