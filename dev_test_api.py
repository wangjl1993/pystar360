
import argparse
from importlib import import_module
import time
from collections import Counter
from datetime import datetime
import torch 
import omegaconf

from uni360detection.base.dataStruct import CarriageInfo, QTrainInfo
from uni360detection.utilities.fileManger import read_yaml

DEFAULT_STR_DATETIME = "%Y-%m-%d-%H-%M-%S%f"

parser = argparse.ArgumentParser()
parser.add_argument('-c',
                    '--config',
                    default="./dev_CRH1A-A.yaml",
                    type=str,
                    help='config')
parser.add_argument('-t',
                    '--types',
                    default="ok",
                    type=str,
                    help='ok, defect')
            
def main():
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config)
    time_str = datetime.now().strftime(DEFAULT_STR_DATETIME)
    fname = args.types + time_str
    
    image_folder = config[args.types]

    major_train_code = config.major_train_code
    minor_train_code = config.minor_train_code
    folder_name = minor_train_code.replace("-", "_")
    pyStar360Robot = getattr(import_module(f"{folder_name}.main"), 'pyStar360Robot')
    final_output = {}
    output  = {}
    s0 = time.time()
    for train_num, train_sn, test_path in image_folder:
        for channel in config.channels:
            
            if isinstance(config.carriages, str):
                idx = list(map(int,config.carriages.split(",")))
                carriages = range(idx[0], idx[1])
            elif isinstance(config.carriages, list) or isinstance(config.carriages, omegaconf.ListConfig):
                carriages = config.carriages
            else:
                raise ValueError("Please either use list or str(range)")
            for carriage in carriages:

                queryTrain = QTrainInfo(
                    major_train_code, minor_train_code, train_num, train_sn, 
                    channel, int(carriage), CarriageInfo())
                queryTrain.test_train.path = test_path
                robot = pyStar360Robot(queryTrain, f"./{folder_name}/channel_params.yaml", 
                        f"./{folder_name}/item_params.yaml", f"./{folder_name}/template/{channel}/template.json",
                        device=torch.device("cuda:2"))
                robot.run()
            #     cutpoints = robot.run()
            #     output[str(carriage)] = {"start_line": cutpoints[0], "end_line": cutpoints[1]}
            
            # final_output["carriage"] = output
            # write_json("./aux_data.json",final_output, mode="w")

if __name__ == "__main__":
    main()