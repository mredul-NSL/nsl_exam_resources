import cv2

root_folder = '/media/nsl4/hdd2/mredul/exam/mnist_synthetic/synthetic_images/'
save_folder = '/media/nsl4/hdd2/mredul/exam/mnist_synthetic/synthetic_images/resized_images/'

for i in range(0,10):
    img = cv2.imread(root_folder + "digit_number_img_" + str(i) + ".jpg")

    new_img = cv2.resize(img, (28, 28))

    new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)

    new_img = 255 - new_img

    #cv2.imshow(str(i), new_img)
    cv2.imwrite(save_folder + str(i)+".jpg", new_img)


