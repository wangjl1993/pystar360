

# import sys 
# from pathlib import Path
# FILE = Path(__file__).absolute()
# # sys.path.append((FILE.parents[1] / "uni360detection").as_posix())
# # sys.path.append(FILE.parents[0].as_posix())

# import sys
# import os 
# # sys.path.append( os.path.join( os.path.dirname(__file__), ".." , ".." )) 
# # print(sys.path)

# if __package__:
#     from .. import config
# else:
#     sys.path.append(str(FILE.parents[0]) + '/..')
#     from uni360detection.base.dataStruct import *

print('__file__={0:<35} | __name__={1:<25} | __package__={2:<25}'.format(__file__,__name__,str(__package__)))
#
# from uni360.uni360detection.base.dataStruct import QTrainInfo
import sys
sys.path.append("..")
from uni360detection.base.dataStruct import QTrainInfo

A  = QTrainInfo()

