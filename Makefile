stest:
	-rm -rf C:/Users/Nick\ LIU/Desktop/winter\ project/DeepTADARData/DeepTADARDataCollection/RawData/exp01
	D:/ProgramData/miniconda/envs/py3.8/python.exe "c:/Users/Nick LIU/Desktop/winter project/DeepTADARData/DeepTADARDataCollection/data_collection.py" --collection_duration 1 --save_data 1 --save_path RawData/exp01
test: 
	D:/ProgramData/miniconda/envs/py3.8/python.exe "c:/Users/Nick LIU/Desktop/winter project/DeepTADARData/DeepTADARDataCollection/visualize.py"

clean:
	-rm -rf C:/Users/Nick\ LIU/Desktop/winter\ project/DeepTADARData/DeepTADARDataCollection/RawData/exp01