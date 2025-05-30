import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# def draw_boxes(img, bbox, identities=None, offset=(0,0),trace={},pre={}):
def draw_boxes(img, bbox, identities=None, offset=(0, 0), trace={}, pre={} ,pre_h={}):
    # pre:{1:[[1,2],[3,4],[4,5],[5,6]],}
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        print("historical trajectory:",trace[id])
        for point in trace[id]:
            cv2.circle(img, point, 3, color, 4)

        if id in pre:
            print("predicted trajectory",pre[id])
            for point in pre[id]:
                if point[0] > 1080:
                    point[0] = 1079
                if point[1] > 1920:
                    point[1] = 1919
                # print(id)
                point = tuple(point)
                # print(point)
                cv2.circle(img, point, 2, (0,0,255), 4)
            # print("historical predictions:", pre_h[id])
            for point in pre_h[id]:
                if point[0] > 1080:
                    point[0] = 1079
                if point[1] > 1920:
                    point[1] = 1919
                # print(id)
                point = tuple(point)
                # print(point)
                cv2.circle(img, point, 2, (0, 0, 255), 4)
    return img
if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
