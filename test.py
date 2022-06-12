from uni360detection.base.dataStruct import *

p = Point(1,2)
r = Rect(*[[122,3],[3,4]])
b = BBox()
# func.proposal_rect = [[0,0], [0,0]]
# r[1][1] = 66666
# print(r)

# r[1][1] = 8777
# print(r.get_area())
# print(r)
# print(r.to_list())


# r[1][1] = 1230985013
# print(r)
# [[0,0], [0,0]]
# print(func.to_tuple())
b.curr_rect = [[2,3],[4,4]]
print(b)
b.curr_rect[0][0] = 10000
print(b)