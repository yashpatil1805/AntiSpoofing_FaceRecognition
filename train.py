from ultralytics import YOLO

model = YOLO('yolov8n.pt')
def main():

    model.train(data='Dataset/SplitData/dataOffline.yaml',epochs=100     )

if __name__== '__main__':
    main()