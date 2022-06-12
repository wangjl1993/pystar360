from uni360detection.base.dataStruct import *

func = Rect()

func[1][0] = 10000000
func[0][1] = 999
print(func)
print(func[0])
print(func[1])
print(func.to_tuple())
print(astuple(func))