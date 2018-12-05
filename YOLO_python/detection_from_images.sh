m="10"

# carrots
python darknet_find_object_from_images.py -id ../../../dataset/carrots/web_images -l carrot -od ../../../dataset/detected_by_gcnn/from_web_images -m $m

# bottles
python darknet_find_object_from_images.py -id ../../../dataset/bottles/web_images/alcohol_glass_bottles_brown -l bottle -od ../../../dataset/bottles/detected_by_gcnn/alcohol_grass_bottles_brown -m $m

python darknet_find_object_from_images.py -id ../../../dataset/bottles/web_images/clear_glass_bottles/ -l bottle -od ../../../dataset/bottles/detected_by_gcnn/clear_glass_bottles/ -m $m

python darknet_find_object_from_images.py -id ../../../dataset/bottles/web_images/brown_and_green_glass_bottles/ -l bottle -od ../../../dataset/bottles/detected_by_gcnn/brown_and_green_glass_bottles -m $m

python darknet_find_object_from_images.py -id ../../../dataset/bottles/web_images/alcohol_glass_bottles_green -l bottle -od ../../../dataset/bottles/detected_by_gcnn/alcohol_grass_bottles_green -m $m

python darknet_find_object_from_images.py -id ../../../dataset/bottles/web_images/alcohol_glass_bottles_clear -l bottle -od ../../../dataset/bottles/detected_by_gcnn/alcohol_grass_bottles_clear -m $m

python darknet_find_object_from_images.py -id ../../../dataset/bottles/web_images/color_glass_bottles -l bottle -od ../../../dataset/bottles/detected_by_gcnn/color_glass_bottles -m $m
