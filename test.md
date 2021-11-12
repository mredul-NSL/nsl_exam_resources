# Evaluation Question for NSL Reaserch Assistants

## Index

- [Instructions](https://github.com/NSLabTeam/nslraevaluation/blob/master/evaluation-nsl-ras.md#instructions)
- [A. Unix commands](https://github.com/NSLabTeam/nslraevaluation/blob/master/evaluation-nsl-ras.md#a-unix-commands)
- [B. Python basic](https://github.com/NSLabTeam/nslraevaluation/blob/master/evaluation-nsl-ras.md#b-python-basic)
- [C. Python OOP](https://github.com/NSLabTeam/nslraevaluation/blob/master/evaluation-nsl-ras.md#c-python-oop)
- [D. Data structure & algorithm](https://github.com/NSLabTeam/nslraevaluation/blob/master/evaluation-nsl-ras.md#d-data-structure--algorithm)
- [E. Numpy](https://github.com/NSLabTeam/nslraevaluation/blob/master/evaluation-nsl-ras.md#e-numpy)
- [F. Deep learning](https://github.com/NSLabTeam/nslraevaluation/blob/master/evaluation-nsl-ras.md#f-deep-learning)

## Instructions

1. Numbers placed at the end of each question represents the points for that specific question/problem.

2. Total time for this evaluation is: `6.5 hr`.

3. Create a directory named `ra_evaluation_<name>`. Write your answers in `.docx` file and use `.py` files where necessary. Put all the files inside the `ra_evaluation_<name>` directory.

4. You need to upload the directory to your NSL github repository immediately after the evaluation time ended.
5. You must answer all the sections of the question

## A. Unix commands

### A1. Let's assume you are working on 1000K of images files. When you open the image directory in your file manager it gets freezed. What shold you do in this situation to open image withing freezing ?  `[2]`

### A2. Write a ssh command that will transfer list of file and folders from one computer to another computer those are connected in same local netwrok `[2]`

### A3. Suppose you have to 10GB single text file. `[3]`

    a. Write a command that shows last 100 lines of text from 10GB text files
      
    b. Find if the keyword "ImageNet" is in the text file and counts it's occurences

### A4. How to check size (disk space taken) of a installed python package `[3]`

### A5. How to check which python is being used in current terminal session `[2]`

## B. Python basic

### B1. Suppose we have two python variables *arg **args like bellow. `[2]`

```python
def method_name(*arg, **args):
    pass
```

what is the differrence between these two veriables in python? 

### B2. We have code like bellow `[2]`

```python
def iterator(start, end):
    pass

iterator(10, 10)
iterator(start=10, 20)
iterator(10, end=10)
iterator(start=10, end=20)
```

What will happen in the above 4 different calls of the same method `iterator()`?

### B3. Suppose we have a string object as following, `[2]`

```python
lab_name = "NSLab"
```

Can we do the bellow operations? Yes or not, explain.

```python
lab_name[0] = 'n'
lab_name[1] = 's'
lab_name[2] = 'l'
lab_name[3] = 'A'
lab_name[4] = 'b'
```

### B4. Suppose you have a method named `get_info()` and it return two values `name` and `address` like bellow- `[3]`

```python
def get_info():
    name = get_name()
    address = get_address()
    return name, address
```

We can get info by calling, 

```python
name, address = get_info()
```

Let's say, we do not need to use variable `address` then what should be the best way to call `get_info()` method?

### B5. Suppose you are given the bellow method for doing a specific mathematical operation. Can you find out the breaking point of the method and how can you fix that if any? `[5]`

```python
def do_math(number1, number2):
    return number1 / number2

# lets call the method with x, y where x and y are two numbers
do_math(x, y)
```

## C. Python OOP

### C1. Suppose you have a class `[9]`

```python
class Person:
    def __init__(self, firstname, lastname):
        self.first = firstname
        self.last = lastname
```

Let's say we have two object like bellow, 

```python
p_1 = Person("Mehadi", "Hasan")
p_2 = Person("Rashed", "Mahbub")
```

Now update the `Person` class code so that we can do the following operation

a. p1 == p2

b. p1 != p2

c. p1 < p2

d. p1 > p2

e. p1 + p2


### C2. We have a class like bellow, `[9]`

```python
class Info: 
    def __inif__(self, first_name, last_name, address, age): 
         self.first_name = first_name 
         self.last_name = last_name 
         self.address = address 
         self.age = age 
  
infos = [ 
     Info("Mehadi", "Hasan", "Bogra", 28), 
     Info("Mehadi", "Hasan", "Bogra", 25), 
     Info("Kousar", "Rahman", "Dhaka", 30) 
]
```

Now we want to sort `infos` list using `sorted` function How can we do it ? 

Sorting criterion is if 1st name is equal for both object then compare with last name
if last name is equal comapre address, if address is equal compare age

## D Data structure & algorithm

### D1 Write a program in python that takes a text paragraph written in English as input and returns a mapping for words and their frequencies in ascending order. If the frequency of two words are same, then the words need to be sorted based on their occurrence in the text.  [discard any symbols or punctuations except a-zA-Z0-9] `[10]`

```python
# Sample Input:
"hello" is the first word written by a programmer in "hello world" program.

# Sample Output:
word1: frequency1
word2: frequency2
word3: frequency3
.................
wordN: frequencyN
```

__NOTE__: Use [this text file](./datasets/pg3807.txt) for your program

### D2  You have the following code `[2]`

```python
s = set(['s', 'p', 'a', 'm'])
l = ['s', 'p', 'a', 'm']

def lookup_set(s):
    return 's' in s

def lookup_list(l):
    return 's' in l
```

Which lookup method (lookup_set, lookup_list) is best ? and why you think it is best ?

### D3 We have following code `[10]`

```python
import random

max_num = 30000

first_list = [random.randint(0, max_num*3) for _ in range(max_num)]
second_list = [random.randint(0, max_num*4) for _ in range(max_num)]

num_list = []
for num in first_list:
    if num in second_list:
        num_list.append(num)
```

It takes around 18 seconds to run. How to optimized the code so that it can run less then 1 seconds

_NOTE: Use $ time python name_of_script.py command to run and calculate time for your program

## E. Numpy

### E1. Let's assume, you have an array or numbers like below- `[6]`

```python
A = [[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11],
[12, 13, 14, 15]]
```

a. Write a statement with slice notation that will select center elements from the array `A`. The resultant array might look like below- 

```python
[[ 5,  6],
[ 9, 10]]
```

b. Pick all values from `A` except the last column. The resultant array might look like below.

```python
[[ 0,  1,  2],
[ 4,  5,  6],
[ 8,  9, 10],
[12, 13, 14]]
```

c. Pick all values from `A` except the first row. The resultant array might look like below.

```python
[[ 4,  5,  6,  7],
[ 8,  9, 10, 11],
[12, 13, 14, 15]]
```

### E2. Suppose you are given a numpy array like bellow-  `[3]`

```python
A = [[[[ 0,  1,  2],
        [ 3,  4,  5]],

        [[ 6,  7,  8],
        [ 9, 10, 11]]],


        [[[12, 13, 14],
        [15, 16, 17]],

        [[18, 19, 20],
        [21, 22, 23]]]]
```

Calculate the row wise sum of the array without using explicit loop.
See the output for clarification-

```python
Row_sum =   [[[ 3, 12],
            [21, 30]],

            [[39, 48],
            [57, 66]]]
```


## F. Deep learning

### F1. Suppose you have a polynomial equation `y = 5x`<sup>`2`</sup>` + 7y + 9`. Now, you want to learn this equation by a neural network. Write the possible solution in python and tensorflow. To be more specefic we have below input and target. `[25]`

__NOETE__: You can not use Dense layer to solve this problme. Your soluton must impliment the polynomial hypothesis explicitly (e.g: You can use `tf.Variable` to define the given polinomial equation) 

```python
x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([5*i**2 + 7*i + 9 for i in x_train])
```

Input / Output:

```python
If we input to our model 0 it should print 9
If we input to our model 1 it should print 21
If we input to our model 2 it should print 43
...
for 9 it should print 477
```

### F2. Solving model overfitting issue `[25]`

You are given a code snipt for training a neural network model for classifying cats and dogs. Run the training in a python environment. It seems that the model will overfit in the learning process. Can you fix the overfitting issue so that the distance in between training accuracy and validation accuracy curve is closer as much as possible.

```python
# Download dataset
wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip
  
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)
```

Model result should look like bellow

```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

![image](./images/overfitting.png)
