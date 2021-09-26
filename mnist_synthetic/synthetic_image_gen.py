from PIL import Image

from PIL import ImageFont
from PIL import ImageDraw

# Plot question mark:


img = Image.new('RGB', (500,500), (250,250,250))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("UbuntuMono-B.ttf", 400)
draw.text((180, -30),"?",(0),font=font)
img.save('question_mark_img.jpg')

# plot digit numbers (from 0 to 9):

for i in range(10):
    img = Image.new('RGB', (500,500), (250,250,250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("UbuntuMono-B.ttf", 480)
    draw.text((150, -30),str(i),(0),font=font)
    img.save('digit_number_img_'+str(i)+'.jpg')