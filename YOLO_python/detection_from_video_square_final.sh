m="15"
l="carrot"

# carrots group_a
for i in `seq 1 28`
do
    echo video: carrot_a$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/final/group_a/carrot_a$i.avi -l $l -od ../../../dataset/carrots/detected_by_gcnn/final/from_videos_square/group_a/carrot_a$i -m $m -square
    sleep 5
done

# carrots group_b
for i in `seq 1 28`
do
    echo video: carrot_b$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/final/group_b/carrot_b$i.avi -l $l -od ../../../dataset/carrots/detected_by_gcnn/final/from_videos_square/group_b/carrot_b$i -m $m -square
    sleep 5
done

