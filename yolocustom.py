from ultralytics import YOLO

def test():
    model=YOLO("models/best.pt")
    result=model.predict("11.jpg",save=False)
    print()

if __name__=='__main__':
    test()