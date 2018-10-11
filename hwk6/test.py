from train import AlexNet
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Directory to the saved model')

args = parser.parse_args()

class TestClass:
    def __init__(self):
        self.model = AlexNet()
        # load from model
        self.check_point_file = os.path.join(args.save, 'alex_checkpoint.tar')
        if not os.path.exists(os.path.dirname(self.check_point_file)):
            try:
                os.makedirs(os.path.dirname(self.check_point_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        if os.path.isfile(self.check_point_file):
            cp = torch.load(self.check_point_file)
            self.start = cp['epoch']
            self.best_acc = cp['best_acc']

            print('checkpoint found at epoch', self.start)
            self.model.load_state_dict(cp['model'])
            self.tiny_class = cp['tiny_class']
            self.classes = cp['classes']
        else:
            print('No model found. Exit!!!')
            exit()

    def forward(self, img):
        _4d = troch.unsqueeze(img.type(torch.FloatTensor), 0)
        self.model.eval()
        output = self.model(_4d)
        _, result = torch.max(output, 1)

        print('predicted', result)
        label = self.tiny_class[self.classes[result.data[0]]]
        return label


    def cam(self, idx = 0):
            
            def prepare(img_origin):
                # Convert to Tensor and Normalize
                transformer = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                
                return prepare(image)

            cam = cv2.VideoCapture(idx)
            cam.set(3, 1280)
            cam.set(4, 720)
            cv2.namedWindow("test")
            img_counter = 0
            print('Press e/E to exit, c/C to capture a picture\n')
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                norm_img_tensor = prepare(frame)
                predicted_category = self.forward(norm_img_tensor)
                print(predicted_category)
                cv2.putText(frame, predicted_category, (10,500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.imshow('Capturing', frame)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('e'):
                    # e pressed
                    print("E hit, closing...")
                    break
                elif k == ord('c'):
                    # c pressed
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1
                
            cam.release()
            cv2.destroyAllWindows()
   
