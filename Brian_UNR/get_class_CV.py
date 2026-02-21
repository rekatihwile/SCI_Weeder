# main.py

# Import the WeedCV class from your other file
# (Change 'weed_cv' to whatever your actual filename is, minus the .py)
from cv_helpers import WeedCV

def main():
    # 1. Define your model paths
    YOLO_WEIGHTS = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/Brian_UNR/weights/yolo_w_kale.pt"
    MOBILENET_WEIGHTS = "/home/laser/Documents/Laser_Workspace/SCI_Weeder/Brian_UNR/weights/sniper.pt"

    print("[INFO] Initializing models... this might take a second.")
    
    # 2. Run the __init__ function by creating an instance of the class
    weeder = WeedCV(
        yolo_path=YOLO_WEIGHTS, 
        mobilenet_path=MOBILENET_WEIGHTS, 
        conf=0.30, 
        iom_thresh=0.80
    )

    # 3. Print the class dictionary to see your Roboflow labels
    print("\n--- YOLO Class Mapping ---")
    print(weeder.yolo.names)
    print("--------------------------\n")

if __name__ == "__main__":
    main()