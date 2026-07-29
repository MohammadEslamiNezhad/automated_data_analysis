@app.post('/upload_file/')
async def upload_file(file: UploadFile = File(...)):
    global data 
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file.file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file.file)
    else: 
        return {'message': 'choose right file type'}

    return {'detail': 'file uploaded successfully'}