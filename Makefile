stest:
	-rm -rf C:/Users/Nick\ LIU/Desktop/winter\ project/DeepTADARData/DeepTADARDataCollection/RawData/exp01
	D:/ProgramData/miniconda/envs/py3.8/python.exe "c:/Users/Nick LIU/Desktop/winter project/DeepTADARData/DeepTADARDataCollection/data_collection.py" --collection_duration 5 --save_data 1 --save_path RawData/exp01
test: 
	D:/ProgramData/miniconda/envs/py3.8/python.exe "c:/Users/Nick LIU/Desktop/winter project/DeepTADARData/DeepTADARDataCollection/visualize.py"

clean:
	-rm -rf C:/Users/Nick\ LIU/Desktop/winter\ project/DeepTADARData/DeepTADARDataCollection/RawData/exp01

visualize: 
	D:/ProgramData/miniconda/envs/py3.8/python.exe "c:/Users/Nick LIU/Desktop/winter project/DeepTADARData/DeepTADARDataCollection/visualize.py"


calib:
	D:/ProgramData/miniconda/envs/py3.8/python.exe calib.py

sepcalib:
	D:/ProgramData/miniconda/envs/py3.8/python.exe depth_layer_sep_calib.py

select:
	D:/ProgramData/miniconda/envs/py3.8/python.exe select_points_for_calib.py 

select1:
	D:/ProgramData/miniconda/envs/py3.8/python.exe select_points_for_calib_and_writeback.py 

mock: 
	D:/ProgramData/miniconda/envs/py3.8/python.exe mock_data_prepare.py

example: 
	D:/ProgramData/miniconda/envs/py3.8/python.exe example.py